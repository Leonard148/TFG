import argparse
import time
import json
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from datetime import datetime
import sys
import traceback

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, \
    scale_coords, xyxy2xywh, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized, TracedModel

def detect_timed_captures(opt):
    """
    Realiza detección de objetos periódicamente en capturas de video
    utilizando un modelo YOLO entrenado.
    """
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')
    
    # Create output directory with timestamp
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(increment_path(Path(opt.project) / f"{opt.name}_{run_timestamp}", exist_ok=opt.exist_ok))
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if save_txt:
        (save_dir / 'labels').mkdir(parents=True, exist_ok=True)

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    try:
        model = attempt_load(weights, map_location=device)
        stride = int(model.stride.max())
        imgsz = check_img_size(imgsz, s=stride)

        if trace:
            model = TracedModel(model, device, opt.img_size)

        if half:
            model.half()  # to FP16
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Initialize camera
    try:
        cap = cv2.VideoCapture(int(source) if source.isnumeric() else source)
        if not cap.isOpened():
            raise IOError(f"No se puede abrir la fuente de video: {source}")
    except Exception as e:
        print(f"Error al inicializar la cámara: {e}")
        return

    # Set camera parameters (optional - uncomment if needed)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # cap.set(cv2.CAP_PROP_FPS, 30)

    # Get actual camera parameters
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Cámara inicializada con resolución {actual_width}x{actual_height} @ {actual_fps} FPS")

    # Warmup model
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    last_capture_time = time.time()
    capture_interval = opt.interval  # seconds between captures
    frame_count = 0
    processed_count = 0

    # Create log file
    log_file = save_dir / "detection_log.txt"
    with open(log_file, 'w') as f:
        f.write(f"Detection Log - Started at {run_timestamp}\n")
        f.write(f"Model: {weights}, Device: {device}\n")
        f.write(f"Camera resolution: {actual_width}x{actual_height} @ {actual_fps} FPS\n")
        f.write("=" * 50 + "\n")

    try:
        print(f"Iniciando captura periódica cada {opt.interval} segundos...")
        while True:
            ret, im0s = cap.read()
            if not ret:
                print("Error al leer el frame. Verificando conexión...")
                # Try to reconnect
                time.sleep(1.0)
                cap.release()
                cap = cv2.VideoCapture(int(source) if source.isnumeric() else source)
                if not cap.isOpened():
                    print("No se pudo reconectar. Saliendo.")
                    break
                continue

            frame_count += 1
            current_time = time.time()
            
            # Only process at the specified interval
            if current_time - last_capture_time >= capture_interval:
                last_capture_time = current_time
                processed_count += 1
                
                # Get precise capture timestamp
                capture_time = datetime.now()
                timestamp_str = capture_time.strftime("%Y-%m-%d_%H-%M-%S.%f")[:-3]  # Include milliseconds
                
                # Process original image
                im0 = im0s.copy()
                
                # Convert image to model input format
                img = cv2.resize(im0s, (imgsz, imgsz))
                img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()
                img /= 255.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                # Inference
                t1 = time_synchronized()
                with torch.no_grad():
                    pred = model(img, augment=opt.augment)[0]
                t2 = time_synchronized()

                # Apply NMS
                pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, 
                                          classes=opt.classes, agnostic=opt.agnostic_nms)
                t3 = time_synchronized()

                # Process detections
                detections = []
                s = ''
                
                # Scale coords back to original image size
                if len(pred[0]) > 0:
                    pred[0][:, :4] = scale_coords(img.shape[2:], pred[0][:, :4], im0.shape).round()
                
                for *xyxy, conf, cls in reversed(pred[0]):
                    c = int(cls)
                    class_name = names[c]
                    confidence = float(conf)
                    
                    # Skip low confidence detections for specific classes if configured
                    if hasattr(opt, 'class_conf_thresh') and c in opt.class_conf_thresh:
                        if confidence < opt.class_conf_thresh[c]:
                            continue
                    
                    label = f'{class_name} {confidence:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors[c], line_thickness=2)
                    
                    # Convert coordinates to [x1, y1, x2, y2] format
                    bbox = [float(x) for x in xyxy]
                    
                    # Add detection to list
                    detections.append({
                        'class': class_name,
                        'class_id': c,
                        'confidence': confidence,
                        'bbox': bbox,
                        # Calculate center for mapping purposes 
                        'center': [(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2],
                    })
                    
                    s += f"{class_name} {confidence:.2f}, "

                # Get GPS coordinates (placeholder - adapt to your GPS hardware)
                gps_coords = get_gps_coordinates() if hasattr(opt, 'use_gps') and opt.use_gps else None

                # Print time and detections
                inference_time = (1E3 * (t2 - t1))
                nms_time = (1E3 * (t3 - t2))
                msg = f'{timestamp_str}: {s}Done. ({inference_time:.1f}ms) Inference, ({nms_time:.1f}ms) NMS'
                print(msg)
                
                # Log to file
                with open(log_file, 'a') as f:
                    f.write(f"{timestamp_str}: {len(detections)} detections. {s}\n")

                # Save results
                if save_img:
                    # Save image with timestamp in filename
                    img_filename = f"capture_{timestamp_str}.jpg"
                    save_path = str(save_dir / img_filename)
                    cv2.imwrite(save_path, im0)
                    
                    # Create metadata dictionary with mapping information
                    metadata = {
                        'filename': img_filename,
                        'timestamp': timestamp_str,
                        'detections': detections,
                        'location': gps_coords,
                        'camera_parameters': {
                            'resolution': [actual_width, actual_height],
                            'fps': actual_fps,
                            # Add other camera calibration parameters if available
                        },
                        'processing_times': {
                            'inference_ms': inference_time,
                            'nms_ms': nms_time
                        }
                    }
                    
                    # Save metadata as JSON
                    metadata_path = str(save_dir / f"metadata_{timestamp_str}.json")
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)

                # Display if requested
                if view_img:
                    # Add timestamp and other info to display image
                    info_text = f"Time: {timestamp_str} | FPS: {1000/(inference_time+nms_time):.1f} | Detections: {len(detections)}"
                    font_scale = min(im0.shape[1] / 1000, 1.0)
                    cv2.putText(im0, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
                    
                    display_img = cv2.resize(im0, (min(1280, im0.shape[1]), min(720, im0.shape[0])))
                    cv2.imshow('YOLO Timed Capture', display_img)
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        break

            # Small delay to prevent maxing out CPU
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("Detección interrumpida por el usuario")
    except Exception as e:
        print(f"Error en la detección: {e}")
        traceback.print_exc()
    finally:
        # Cleanup
        cap.release()
        if view_img:
            cv2.destroyAllWindows()
        
        # Final statistics
        elapsed_time = time.time() - (last_capture_time - capture_interval)
        print(f'Finalizado. Capturas procesadas: {processed_count}/{frame_count} frames')
        print(f'Tiempo total: {elapsed_time:.1f} segundos')
        print(f'Resultados guardados en: {save_dir}')

def get_gps_coordinates():
    """
    Placeholder function for GPS coordinate retrieval.
    Replace with actual implementation for your GPS hardware.
    """
    # Example return format compatible with mapping applications
    return {
        "latitude": 0.0,  # Replace with actual GPS read
        "longitude": 0.0, # Replace with actual GPS read
        "altitude": 0.0,  # Replace with actual GPS read if available
        "accuracy": 0.0,  # GPS accuracy in meters if available
        "timestamp": datetime.now().isoformat()
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='0', help='source (0 for webcam)')
    parser.add_argument('--img-size', '--img', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help="don't trace model")
    parser.add_argument('--interval', type=float, default=2.0, help='interval between captures in seconds')
    parser.add_argument('--use-gps', action='store_true', help='enable GPS coordinate logging')
    
    opt = parser.parse_args()
    print(f"Opciones de ejecución: {opt}")
    
    with torch.no_grad():
        detect_timed_captures(opt)