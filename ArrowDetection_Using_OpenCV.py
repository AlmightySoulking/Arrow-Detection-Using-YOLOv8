
import cv2
import numpy as np
import math
from collections import deque
import time

class ArrowDetector:
    def __init__(self):
        """Initialize the Arrow Detector with configurable parameters"""
        # Detection parameters
        self.min_contour_area = 500
        self.max_contour_area = 50000
        self.epsilon_factor = 0.02  # For contour approximation
        self.angle_threshold = 30   # Degrees for angle detection

        # Direction tracking
        self.direction_history = deque(maxlen=10)
        self.confidence_threshold = 0.6

        # Color ranges for arrow detection (HSV)
        # Black arrows
        self.lower_black = np.array([0, 0, 0])
        self.upper_black = np.array([180, 255, 80])

        # Alternative: detect any dark objects
        self.lower_gray = np.array([0, 0, 0])
        self.upper_gray = np.array([180, 50, 100])

    def preprocess_frame(self, frame):
        """Preprocess the frame for better arrow detection"""
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create mask for dark objects (arrows)
        mask1 = cv2.inRange(hsv, self.lower_black, self.upper_black)
        mask2 = cv2.inRange(hsv, self.lower_gray, self.upper_gray)
        mask = cv2.bitwise_or(mask1, mask2)

        # Morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Apply Gaussian blur to reduce noise
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        return mask

    def find_arrow_tip(self, contour):
        """Find the tip of the arrow using convex hull analysis"""
        # Get convex hull
        hull = cv2.convexHull(contour, returnPoints=False)

        # Find convexity defects
        if len(hull) > 3:
            defects = cv2.convexityDefects(contour, hull)

            if defects is not None:
                # Find the point with maximum distance from hull
                max_distance = 0
                tip_point = None

                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    far = tuple(contour[f][0])

                    if d > max_distance:
                        max_distance = d
                        tip_point = far

                return tip_point

        # Fallback: find the point farthest from centroid
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroid = (cx, cy)

            max_dist = 0
            tip_point = None

            for point in contour:
                x, y = point[0]
                dist = math.sqrt((x - cx)**2 + (y - cy)**2)
                if dist > max_dist:
                    max_dist = dist
                    tip_point = (x, y)

            return tip_point

        return None

    def analyze_arrow_direction(self, contour):
        """Analyze the direction of the arrow"""
        # Approximate contour to polygon
        epsilon = self.epsilon_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Need at least 4 points for an arrow
        if len(approx) < 4:
            return None, 0, None

        # Find arrow tip
        tip_point = self.find_arrow_tip(contour)

        if tip_point is None:
            return None, 0, None

        # Calculate centroid
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None, 0, None

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centroid = (cx, cy)

        # Calculate direction vector from centroid to tip
        dx = tip_point[0] - cx
        dy = tip_point[1] - cy

        # Calculate angle in degrees
        angle = math.degrees(math.atan2(dy, dx))

        # Convert to compass directions
        direction = self.angle_to_direction(angle)

        # Calculate confidence based on arrow shape analysis
        confidence = self.calculate_confidence(contour, approx)

        return direction, angle, tip_point, confidence, centroid

    def angle_to_direction(self, angle):
        """Convert angle to compass direction"""
        # Normalize angle to 0-360 range
        angle = (angle + 360) % 360

        directions = {
            (337.5, 360): "East", (0, 22.5): "East",
            (22.5, 67.5): "Northeast",
            (67.5, 112.5): "North",
            (112.5, 157.5): "Northwest",
            (157.5, 202.5): "West",
            (202.5, 247.5): "Southwest",
            (247.5, 292.5): "South",
            (292.5, 337.5): "Southeast"
        }

        for (start, end), direction in directions.items():
            if start <= angle < end:
                return direction

        return "Unknown"

    def calculate_confidence(self, contour, approx):
        """Calculate confidence score for arrow detection"""

        area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)

        # Convexity ratio
        if hull_area > 0:
            convexity = area / hull_area
        else:
            convexity = 0

        # Aspect ratio
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 0

        # Vertex count (arrows typically have 5-7 vertices)
        vertex_score = 1.0 if 4 <= len(approx) <= 8 else 0.5

        # Combine factors
        confidence = (convexity * 0.4 + aspect_ratio * 0.3 + vertex_score * 0.3)

        return min(confidence, 1.0)

    def detect_arrows(self, frame):
        """Main detection function"""
        # Preprocess frame
        mask = self.preprocess_frame(frame)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_arrows = []

        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by area
            if self.min_contour_area < area < self.max_contour_area:
                result = self.analyze_arrow_direction(contour)

                if result[0] is not None:  # Valid direction found
                    direction, angle, tip_point, confidence, centroid = result

                    if confidence > self.confidence_threshold:
                        detected_arrows.append({
                            'contour': contour,
                            'direction': direction,
                            'angle': angle,
                            'tip_point': tip_point,
                            'centroid': centroid,
                            'confidence': confidence,
                            'area': area
                        })

        return detected_arrows, mask

    def draw_results(self, frame, arrows, mask):
        """Draw detection results on frame"""
        result_frame = frame.copy()

        for arrow in arrows:
            contour = arrow['contour']
            direction = arrow['direction']
            angle = arrow['angle']
            tip_point = arrow['tip_point']
            centroid = arrow['centroid']
            confidence = arrow['confidence']

            # Draw contour
            cv2.drawContours(result_frame, [contour], -1, (0, 255, 0), 2)

            # Draw tip point
            cv2.circle(result_frame, tip_point, 5, (0, 0, 255), -1)

            # Draw centroid
            cv2.circle(result_frame, centroid, 3, (255, 0, 0), -1)

            # Draw direction arrow
            arrow_length = 50
            end_x = int(centroid[0] + arrow_length * math.cos(math.radians(angle)))
            end_y = int(centroid[1] + arrow_length * math.sin(math.radians(angle)))
            cv2.arrowedLine(result_frame, centroid, (end_x, end_y), (255, 255, 0), 3)

            # Add text information
            text_x, text_y = centroid[0] + 20, centroid[1] - 20
            cv2.putText(result_frame, f"{direction}", (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(result_frame, f"Angle: {angle:.1f}°", (text_x, text_y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(result_frame, f"Conf: {confidence:.2f}", (text_x, text_y + 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return result_frame


def main():
    """Main function to run live arrow detection"""
    # Initialize camera
    cap = cv2.VideoCapture(0)  # Use 0 for default camera

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Initialize detector
    detector = ArrowDetector()

    # Performance tracking
    fps_counter = 0
    fps_start_time = time.time()

    print("Starting live arrow detection...")
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save current frame")
    print("  'c' - Toggle color detection mode")
    print("  'SPACE' - Pause/Resume")

    paused = False
    color_mode = 0  # 0: black arrows, 1: any dark objects

    while True:
        if not paused:
            ret, frame = cap.read()

            if not ret:
                print("Error: Could not read frame")
                break

        # Detect arrows
        arrows, mask = detector.detect_arrows(frame)

        # Draw results
        result_frame = detector.draw_results(frame, arrows, mask)

        # Add status information
        status_text = f"Arrows detected: {len(arrows)}"
        cv2.putText(result_frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if paused:
            cv2.putText(result_frame, "PAUSED - Press SPACE to resume", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Calculate and display FPS
        fps_counter += 1
        if fps_counter % 10 == 0:
            current_time = time.time()
            fps = 10 / (current_time - fps_start_time)
            fps_start_time = current_time

        if 'fps' in locals():
            cv2.putText(result_frame, f"FPS: {fps:.1f}", (10, result_frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Show frames
        cv2.imshow('Live Arrow Detection', result_frame)
        #cv2.imshow('Mask', mask)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save current frame
            timestamp = int(time.time())
            cv2.imwrite(f'arrow_detection_{timestamp}.jpg', result_frame)
            print(f"Frame saved as arrow_detection_{timestamp}.jpg")
        elif key == ord('c'):
            # Toggle color detection mode
            color_mode = 1 - color_mode
            if color_mode == 0:
                print("Switched to black arrow detection")
            else:
                print("Switched to dark object detection")
        elif key == ord(' '):
            # Toggle pause
            paused = not paused
            print("Paused" if paused else "Resumed")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Arrow detection stopped")


if __name__ == "__main__":
    main()
