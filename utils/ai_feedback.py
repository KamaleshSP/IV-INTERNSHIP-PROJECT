import pyttsx3
import threading
import queue
import time

class AIFeedback:
    def __init__(self):
        self.tts_engine = None
        self.speech_queue = queue.Queue()
        self.is_speaking = False
        self.speech_thread = None
        self.stop_speech = False
        self.last_spoken_status = None
        self.last_speech_time = 0
        self.min_speech_interval = 3.0  
        
        self.initialize_tts()
        self.start_speech_worker()
        
        # Enhanced status messages with all possible statuses
        self.status_messages = {
            "Active": [
                "Great! You're staying focused.",
                "Good attention level detected.",
                "Keep up the good work!",
                "Excellent focus maintained."
            ],
            "Yawning": [
                "I notice you're yawning. Try to stay alert.",
                "You seem tired. Take a deep breath.",
                "Yawning detected. Please stay focused.",
                "Feeling sleepy? Try to stay awake."
            ],
            "Drowsy": [
                "You appear drowsy. Please stay awake.",
                "Low eye activity detected. Please focus.",
                "Drowsiness alert! Please pay attention.",
                "Your eyes seem heavy. Please stay alert."
            ],
            "Inactive (Face Missing)": [
                "I can't see your face. Please position yourself properly.",
                "Face not detected. Please come back to the camera.",
                "Please ensure you're visible to the camera.",
                "Please return to your position."
            ],
            "Face Missing": [
                "I can't see your face. Please position yourself properly.",
                "Face not detected. Please come back to the camera.",
                "Please ensure you're visible to the camera.",
                "Please return to your position."
            ],
            "Multiple Persons Detected": [
                "Multiple people detected. Please ensure only one person is monitoring.",
                "Too many faces in view. Please clear the area.",
                "Single user mode required.",
                "Only one person should be in view."
            ],
            "Multiple Faces": [
                "Multiple people detected. Please ensure only one person is monitoring.",
                "Too many faces in view. Please clear the area.",
                "Single user mode required.",
                "Only one person should be in view."
            ],
            "Looking Away": [
                "I notice you're looking away. Please focus on your studies.",
                "Please look at the screen.",
                "Your attention seems to be elsewhere.",
                "Please redirect your attention to your work."
            ],
            "Distracted": [
                "You seem distracted. Please focus.",
                "Please pay attention to your studies.",
                "Your attention level is low.",
                "Please concentrate on your work."
            ],
            "Not Awake": [
                "You've been away too long. Please return.",
                "Extended absence detected. Please come back.",
                "Long inactivity period. Please focus.",
                "Please wake up and return to your studies."
            ],
            "Fake Presence": [
                "Artificial presence detected. Please use live video only.",
                "Static image detected. Please show live movement.",
                "Spoofing attempt detected.",
                "Please ensure live video feed."
            ],
            "Emergency": [
                "Emergency alert! Please wake up immediately!",
                "Attention required! You've been inactive too long!",
                "Wake up! Please return to your studies!",
                "Critical alert! Please focus on your work!"
            ],
            "Inactive": [
                "You seem inactive. Please engage with your studies.",
                "No activity detected. Please stay focused.",
                "Please remain active and attentive.",
                "Activity level is low. Please concentrate."
            ]
        }
        
    def initialize_tts(self):
        """Initialize the text-to-speech engine"""
        try:
            self.tts_engine = pyttsx3.init()
            
            # Configure TTS settings
            voices = self.tts_engine.getProperty('voices')
            if voices:
                # Try to use a female voice if available
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
                        
            # Set speech rate and volume
            self.tts_engine.setProperty('rate', 180)  # Words per minute
            self.tts_engine.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)
            
            print("TTS engine initialized successfully")
            
        except Exception as e:
            print(f"Error initializing TTS engine: {e}")
            self.tts_engine = None
            
    def start_speech_worker(self):
        """Start the background thread for speech processing"""
        self.speech_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.speech_thread.start()
        
    def _speech_worker(self):
        """Background worker to process speech queue"""
        while not self.stop_speech:
            try:
                # Get speech request from queue with timeout
                speech_data = self.speech_queue.get(timeout=1)
                
                if speech_data and self.tts_engine:
                    speech_text, priority = speech_data
                    self.is_speaking = True
                    print(f"[AI FEEDBACK] Speaking: {speech_text}")
                    
                    # Clear any pending speech for high priority messages
                    if priority == "high":
                        self.tts_engine.stop()
                    
                    self.tts_engine.say(speech_text)
                    self.tts_engine.runAndWait()
                    
                    self.is_speaking = False
                    
                self.speech_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in speech worker: {e}")
                self.is_speaking = False
                
    def speak_status(self, status):
        """Speak a status-specific message with improved logic"""
        if not self.tts_engine:
            return
            
        current_time = time.time()
        
        # Check if we should speak (avoid too frequent repetition)
        if (self.last_spoken_status == status and 
            current_time - self.last_speech_time < self.min_speech_interval):
            return
            
        # Don't queue new speech if already speaking, unless it's critical
        critical_statuses = ["Emergency", "Not Awake", "Fake Presence", "Drowsy"]
        
        if self.is_speaking and status not in critical_statuses:
            return
            
        # Get appropriate message for status
        messages = self.status_messages.get(status, [f"Status update: {status}"])
        
        # Rotate through messages to avoid repetition
        if len(messages) > 1:
            import random
            message = random.choice(messages)
        else:
            message = messages[0]
        
        # Determine priority
        priority = "high" if status in critical_statuses else "normal"
        
        # Clear queue for critical messages
        if priority == "high":
            self._clear_speech_queue()
            
        # Add to speech queue
        try:
            self.speech_queue.put((message, priority), block=False)
            self.last_spoken_status = status
            self.last_speech_time = current_time
        except queue.Full:
            pass  # Queue is full, skip this message
            
    def speak_custom_message(self, message, priority="normal"):
        """Speak a custom message"""
        if not self.tts_engine:
            return
            
        try:
            self.speech_queue.put((message, priority), block=False)
        except queue.Full:
            pass
            
    def speak_emergency_message(self):
        """Speak emergency wake-up message with high priority"""
        emergency_messages = [
            "Wake up! You've been inactive for too long!",
            "Attention! Please focus on your studies!",
            "Alert! Student attentiveness required!",
            "Emergency! Please return to your work!"
        ]
        
        # Clear queue and speak immediately
        self._clear_speech_queue()
        
        for message in emergency_messages:
            if self.stop_speech:
                break
            try:
                self.speech_queue.put((message, "high"), block=False)
                time.sleep(1.5)  # Pause between messages
            except queue.Full:
                break
                
    def speak_head_pose_feedback(self, head_direction, attention_score):
        """Provide feedback based on head pose"""
        if attention_score < 50:
            if head_direction == "Left":
                message = "Please look forward, not to the left."
            elif head_direction == "Right":
                message = "Please look forward, not to the right."
            elif head_direction == "Up":
                message = "Please look at your screen, not upward."
            elif head_direction == "Down":
                message = "Please look up at your screen."
            else:
                message = "Please focus your attention forward."
                
            self.speak_custom_message(message, "normal")
                
    def _clear_speech_queue(self):
        """Clear all pending speech requests"""
        while not self.speech_queue.empty():
            try:
                self.speech_queue.get_nowait()
                self.speech_queue.task_done()
            except queue.Empty:
                break
                
    def stop_all_speech(self):
        """Stop all speech immediately"""
        self.stop_speech = True
        self._clear_speech_queue()
        
        if self.tts_engine:
            try:
                self.tts_engine.stop()
            except:
                pass
                
    def is_currently_speaking(self):
        """Check if TTS is currently speaking"""
        return self.is_speaking
        
    def get_queue_size(self):
        """Get current speech queue size"""
        return self.speech_queue.qsize()
        
    def reset_speech_timing(self):
        """Reset speech timing to allow immediate feedback"""
        self.last_speech_time = 0
        self.last_spoken_status = None
        
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.stop_all_speech()