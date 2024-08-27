from flask import Flask, send_from_directory
import os
import time

app = Flask(__name__)

# 이미지 파일이 저장된 디렉토리 설정
IMAGE_DIR = '/home/aisw/static'

@app.route('/')
def index():
    return '''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Image Rotator</title>
        </head>
        <body>
            <h1>Image Rotator</h1>
            <img id="image" src="/video_feed" width="640" height="480">
            <script>
                const images = ['/video_feed?t=1', '/video_feed?t=2', '/video_feed?t=3'];
                let index = 0;

                function rotateImage() {
                    index = (index + 1) % images.length;
                    document.getElementById('image').src = images[index] + '?t=' + new Date().getTime();
                }

                setInterval(rotateImage, 2000); // 이미지 교체 주기 (2초)
            </script>
        </body>
        </html>
    '''

@app.route('/video_feed')
def video_feed():
    filename = 'image' + str(int(time.time()) % 3 + 1) + '.jpg'  # 이미지 이름 생성
    return send_from_directory(IMAGE_DIR, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
