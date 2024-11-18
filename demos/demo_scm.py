import cv2
import numpy as np


class FireDetector:
    def __init__(self, alpha=0.5, threshold=25, min_area=500):
        # Background subtraction parameters
        self.alpha = alpha
        self.threshold = threshold
        self.min_area = min_area

        # Background model (mean and variance)
        self.background = None
        self.variance = None

    def initialize_background(self, frame):
        self.background = frame.astype(float)
        self.variance = np.ones_like(frame) * 255.0

    def update_background(self, frame):
        # Update mean
        self.background = self.alpha * frame + (1.0 - self.alpha) * self.background
        # Update variance
        diff = frame - self.background
        self.variance = self.alpha * (diff ** 2) + (1.0 - self.alpha) * self.variance

    def detect_fire(self, frame):
        if self.background is None:
            self.initialize_background(frame)
            return np.zeros_like(frame[:, :, 0]), []

        self.update_background(frame)

        # Background subtraction
        foreground_mask = cv2.absdiff(frame.astype(float), self.background) > (
                self.threshold + 2 * np.sqrt(self.variance))
        foreground_mask = np.bitwise_or.reduce(foreground_mask, axis=2).astype(np.uint8) * 255

        # Fire color filtering
        fire_mask = self.color_filter(frame)

        # Combine foreground and fire color masks
        combined_mask = cv2.bitwise_and(foreground_mask, fire_mask)

        # Morphological operations to remove small noise
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        # Find contours for fire regions
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        fire_regions = [cnt for cnt in contours if cv2.contourArea(cnt) > self.min_area]

        return combined_mask, fire_regions

    def color_filter(self, frame):
        # Fire color model conditions
        B, G, R = cv2.split(frame)
        fire_mask = np.zeros_like(R)

        cond1 = R > G
        cond2 = G > B
        cond3 = (0.25 <= (G / (R + 1))) & ((G / (R + 1)) <= 0.65)
        cond4 = (0.05 <= (B / (R + 1))) & ((B / (R + 1)) <= 0.45)
        cond5 = (0.2 <= (B / (G + 1))) & ((B / (G + 1)) <= 0.6)

        fire_mask[cond1 & cond2 & cond3 & cond4 & cond5] = 255

        return fire_mask


def main():
    video_capture = cv2.VideoCapture('/media/manu/ST2000DM005-2U91/fire/data/20240806/BOSH-FM数据采集/zheng-shinei/Z-D-30m-002.mp4')

    fire_detector = FireDetector()

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Resize the frame to speed up processing
        frame = cv2.resize(frame, (320, 240))

        fire_mask, fire_regions = fire_detector.detect_fire(frame)

        # Draw fire regions
        for region in fire_regions:
            x, y, w, h = cv2.boundingRect(region)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.imshow('Fire Detection', frame)
        cv2.imshow('Fire Mask', fire_mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
