import cv2
import numpy as np

class CourtLineDetectorClassical:
    def __init__(self, reference_config):
        self.reference_config = reference_config

    def process_image(self, image):
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Threshold the grayscale image to extract white pixels
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        return binary

    def detect_lines(self, image):
        # Use the Hough Transform to detect lines in the image
        lines = cv2.HoughLinesP(image, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)

        return lines

    def classify_lines(self, lines):
        # Classify the detected lines as horizontal or vertical
        horizontal_lines = []
        vertical_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < abs(x2 - x1):
                horizontal_lines.append(line)
            else:
                vertical_lines.append(line)

        return horizontal_lines, vertical_lines

    def calculate_homography(self, lines):
        # Calculate the homography matrix based on the intersection points of the detected lines
        src_pts = np.float32([lines[0][0], lines[1][0], lines[2][0], lines[3][0]])
        dst_pts = np.float32(self.reference_config)

        homography_matrix, _ = cv2.findHomography(src_pts, dst_pts)

        return homography_matrix

    def warp_and_count_overlaps(self, image, homography_matrix):
        # Warp the reference court lines using the homography matrix
        warped_image = cv2.warpPerspective(image, homography_matrix, (image.shape[1], image.shape[0]))

        # Count the overlaps with the white pixels of the binary image
        overlaps = np.sum(warped_image == 255)

        return overlaps