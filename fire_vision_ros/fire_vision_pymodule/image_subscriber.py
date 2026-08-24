import time

import cv2
import numpy as np
import rclpy
import subprocess

from rclpy.node import Node
from sensor_msgs.msg import CompressedImage


class ImageSubscriber(Node):

    def __init__(self):
        super().__init__('image_subscriber')

        subprocess.run([
            'v4l2-ctl',
            '-d', '/dev/video0',
            '--set-ctrl=brightness=128'
        ])
        
        self.frame_count = 0
        self.start_time = time.monotonic()

        self.subscription = self.create_subscription(
            CompressedImage,
            '/image_raw/compressed',
            self.image_callback,
            10
        )

        self.get_logger().info(
            'Compressed image subscriber started: /image_raw/compressed'
        )

    def image_callback(self, msg):

        # CompressedImage -> NumPy
        np_arr = np.frombuffer(
            msg.data,
            dtype=np.uint8
        )

        # JPEG -> OpenCV image
        frame = cv2.imdecode(
            np_arr,
            cv2.IMREAD_COLOR
        )

        if frame is None:
            self.get_logger().warning('Decode failed')
            return

        # แสดงภาพ
        cv2.imshow('C130 Camera', frame)
        cv2.waitKey(1)

        # วัด FPS หลัง Decode
        self.frame_count += 1
        elapsed = time.monotonic() - self.start_time

        if elapsed >= 1.0:

            fps = self.frame_count / elapsed

            height, width = frame.shape[:2]

            self.get_logger().info(
                f'Decoded FPS: {fps:.2f} | '
                f'{width}x{height}'
            )

            self.frame_count = 0
            self.start_time = time.monotonic()


def main(args=None):

    rclpy.init(args=args)

    node = ImageSubscriber()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        cv2.destroyAllWindows()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
