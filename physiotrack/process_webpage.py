from flask import Flask, Response
import cv2
from threading import Thread
from queue import Queue

# Replace OpenCV GUI window with a browser-based MJPEG stream
# Usage: call start_webpage_stream() instead of cv2.namedWindow/imshow

def start_webpage_stream():
    """
    Start a Flask server that streams frames to `/video_feed`,
    so you can view real-time pose results in a browser instead of an OpenCV window.
    """
    app = Flask(__name__)

    # queue to receive processed frames
    frame_queue = Queue(maxsize=1)

    def gen_frames():
        # yield frames from process.py via frame_queue
        while True:
            frame = frame_queue.get()
            if frame is None:
                break
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    @app.route('/video_feed')
    def video_feed():
        return Response(gen_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    # start Flask app in background thread
    def run():
        app.run(host='0.0.0.0', port=5000, threaded=True)

    Thread(target=run, daemon=True).start()
    return frame_queue