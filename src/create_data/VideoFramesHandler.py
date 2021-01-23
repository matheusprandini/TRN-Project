import cv2
    
class VideoFramesHandler():

    ## Read video file and return all frames
    @staticmethod
    def extract_frames(filepath, imageSize=(224,224)):

        # Retrieves video information
        video = cv2.VideoCapture(filepath)
        frames = []

        try:
            while True:
                # Get current state
                ret, frame = video.read()
                
                if not ret:
                    break

                # Resized frame
                frame = cv2.resize(frame, imageSize)

                # Normalized frame
                frame = frame / 255.0
                
                # Append frame
                frames.append(frame)
        finally:
            video.release()
            
        return frames