import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from ultralytics import YOLO 

cap = cv2.VideoCapture(0)

class YoloPublisher(Node):

    def __init__(self):
        super().__init__('yolo_publisher')
        self.object_pub = self.create_publisher(String, 'object', 10)
        timer = 0.1 #sec
        self.timer = self.create_timer(timer, self.timer__callback)
        self.model = YOLO("best(1).onnx")

    def timer__callback(self):
        succes, frame = cap.read()

        if succes:
            results = self.model(frame)[0]
            annotated_frame = results.plot()

            msg = String()
            object_detected = results.boxes.cls
            for object in object_detected:
                object_name = results.names[object.item()]
                if object_name == 'person':
                    print("person detected")
                    msg.data = object_name
                    self.object_pub.publish(msg)


            cv2.imshow("YOLO", annotated_frame)
            cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    yolo_publisher = YoloPublisher()
    rclpy.spin(yolo_publisher)
    yolo_publisher.destroy_node()
    rclpy.shutdown()
    

if __name__ == '__main__' :
    main()