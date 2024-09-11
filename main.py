import cv2
import numpy as np
from PIL import Image

def overlay_image_alpha(img, img_overlay, pos):
    """Overlay img_overlay on top of img at the position specified by pos."""
    x, y = pos
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    overlay_image = img_overlay[y1-y:y2-y, x1-x:x2-x]
    img_crop = img[y1:y2, x1:x2]

    alpha = overlay_image[:, :, 3] / 255.0
    alpha_inv = 1.0 - alpha

    for c in range(0, 3):
        img_crop[:, :, c] = alpha * overlay_image[:, :, c] + alpha_inv * img_crop[:, :, c]

    img[y1:y2, x1:x2] = img_crop
    return img

def load_hat_image(filename):
    """Load an image file and convert it to a format suitable for overlay."""
    img = Image.open(filename).convert("RGBA")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)

def create_sidebar(hats_images, current_hat, frame_height):
    sidebar = np.zeros((frame_height, 100, 3), dtype=np.uint8)
    thumbnail_height = frame_height // len(hats_images)
    for i, img in enumerate(hats_images):
        y = i * thumbnail_height
        resized = cv2.resize(img[:,:,:3], (80, thumbnail_height - 20))
        sidebar[y+10:y+thumbnail_height-10, 10:90] = resized
        if i == current_hat:
            cv2.rectangle(sidebar, (5, y+5), (95, y+thumbnail_height-5), (0, 255, 0), 2)
    return sidebar

def get_hat_position(x, y, w, h, hat_index, hat_width):
    """Calculate the position for placing the hat based on the hat type."""
    adjustments = {
        0: (int(w / 8), int(-h / 2)),
        1: (int(w / 12), int(-h / 1.2)),
        2: (int(w / 14), int(-h / 0.8)),
        3: (int(w / 14), int(-h / 1.5)),
        4: (int(w / -11), int(-h / 2)),
        5: (int(w / -199), int(-h / 1.3)),
        6: (int(w / 12), int(-h / 1.1))
    }

    x_offset, y_offset = adjustments.get(hat_index, (0, 0))

    if hat_index == 1:
        y_offset -= int(h / 8)
    elif hat_index == 2:
        hat_width = int(hat_width * 1)
    elif hat_index == 6:
        x_offset -= int(w / 10)
        y_offset -= int(h / 8)
    elif hat_index == 0:
        hat_width = int(hat_width * 1.2)

    return (x + int(w/2) - int(hat_width/2) + x_offset, y + y_offset)

def main():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    hats_images = [
        load_hat_image("hats/gorrohat.png"),
        load_hat_image("hats/mexicanhat.png"),
        load_hat_image("hats/partyhat.png"),
        load_hat_image("hats/pinkhat.png"),
        load_hat_image("hats/redhat.png"),
        load_hat_image("hats/santahat.png"),
        load_hat_image("hats/wizardhat.png")
    ]
    current_hat = 0

    def mouse_callback(event, x, y, flags, param):
        nonlocal current_hat
        if event == cv2.EVENT_LBUTTONDOWN:
            if x > frame.shape[1]:
                clicked_hat = y // (frame.shape[0] // len(hats_images))
                if clicked_hat < len(hats_images):
                    current_hat = clicked_hat

    cv2.namedWindow('Hat Try-On App')
    cv2.setMouseCallback('Hat Try-On App', mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        sidebar = create_sidebar(hats_images, current_hat, frame.shape[0])
        combined_frame = np.hstack((frame, sidebar))

        for (x, y, w, h) in faces:
            if current_hat == 6:
                hat_height = int(h * 1.3)
            elif current_hat == 2 and current_hat == 1:
                hat_height = int(h * 1.5)
            else:
                hat_height = int(h * 1)
            hat_width = int(hat_height * hats_images[current_hat].shape[1] / hats_images[current_hat].shape[0])
            hat = cv2.resize(hats_images[current_hat], (hat_width, hat_height))
            hat_pos = get_hat_position(x, y, w, h, current_hat, hat_width)
            hat_pos = (max(0, hat_pos[0]), max(0, hat_pos[1]))
            
            if hat.shape[0] > 0 and hat.shape[1] > 0:
                frame = overlay_image_alpha(frame, hat, hat_pos)

        combined_frame = np.hstack((frame, sidebar))
        cv2.imshow('Hat Try-On App', combined_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            current_hat = (current_hat + 1) % len(hats_images)
        elif key == ord('p'):
            current_hat = (current_hat - 1) % len(hats_images)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()