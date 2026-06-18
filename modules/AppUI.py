from queue import Empty, Queue
from threading import Thread

import cv2
import numpy as np

try:
    from PySide6.QtCore import QRectF, QTimer
    from PySide6.QtGui import QAction, QColor, QImage, QPainter
    from PySide6.QtOpenGLWidgets import QOpenGLWidget
    from PySide6.QtWidgets import (
        QApplication,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMenu,
        QPushButton,
        QPlainTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:
    raise ImportError(
        "AppUI now uses PySide6. Install it with: pip install PySide6"
    ) from exc


class _Window(QMainWindow):
    def __init__(self, owner, title):
        super().__init__()
        self.owner = owner
        self.setWindowTitle(title)

    def closeEvent(self, event):
        if not self.owner._closing:
            self.owner.close()
        event.accept()


class _GpuImageWidget(QOpenGLWidget):
    def __init__(self, min_size):
        super().__init__()
        self._image = None
        self.setMinimumSize(*min_size)

    def set_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        bytes_per_line = width * 3
        self._image = QImage(
            rgb.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        ).copy()
        self.update()

    def paintGL(self):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("black"))

        if self._image is not None:
            target = self._scaled_target_rect(self._image.width(), self._image.height())
            painter.drawImage(target, self._image)

        painter.end()

    def _scaled_target_rect(self, image_width, image_height):
        widget_width = self.width()
        widget_height = self.height()
        scale = min(widget_width / image_width, widget_height / image_height)
        width = image_width * scale
        height = image_height * scale
        x = (widget_width - width) / 2
        y = (widget_height - height) / 2
        return QRectF(x, y, width, height)


class AppUI:
    def __init__(
        self,
        title="Drone Heatmap",
        on_submit=None,
        on_mask_change=None,
    ):
        self.on_submit = on_submit
        self.on_mask_change = on_mask_change

        self.app = QApplication.instance() or QApplication([])
        self._closing = False

        self.root = _Window(self, title)
        self.graph_window = _Window(self, "Graph")

        self.image_label = _GpuImageWidget((640, 360))
        self.graph_label = _GpuImageWidget((480, 360))

        self.chat_log = QPlainTextEdit()
        self.chat_log.setReadOnly(True)
        self.chat_log.setMaximumBlockCount(1000)
        self.chat_log.setFixedHeight(150)

        self.status_label = QLabel("Ready")

        self.mask_vars = {}
        self._mask_options_initialized = False
        self.mask_menu_button = QPushButton("Observed masks")
        self.mask_menu = QMenu(self.mask_menu_button)
        self.mask_menu_button.setMenu(self.mask_menu)

        self.input_box = QLineEdit()
        self.input_box.returnPressed.connect(self._handle_submit)
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self._handle_submit)

        input_layout = QHBoxLayout()
        input_layout.addWidget(self.input_box, 1)
        input_layout.addWidget(self.send_button)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label, 1)
        layout.addWidget(self.chat_log)
        layout.addWidget(self.status_label)
        layout.addWidget(self.mask_menu_button)
        layout.addLayout(input_layout)

        root_widget = QWidget()
        root_widget.setLayout(layout)
        self.root.setCentralWidget(root_widget)

        graph_layout = QVBoxLayout()
        graph_layout.addWidget(self.graph_label, 1)
        graph_widget = QWidget()
        graph_widget.setLayout(graph_layout)
        self.graph_window.setCentralWidget(graph_widget)

        self._running = False
        self._frame_queue = Queue(maxsize=1)
        self._chat_queue = Queue()
        self._worker = None
        self._poll_timer = QTimer()
        self._poll_timer.timeout.connect(self._poll_frame)

        self._ui_recorder = None
        self._ui_recording_path = None
        self._graph_recorder = None
        self._graph_recording_path = None

        self.root.resize(1400, 950)
        self.graph_window.resize(700, 520)
        self.root.show()
        self.graph_window.show()

    def _handle_submit(self):
        text = self.input_box.text().strip()
        if not text:
            return

        self.status_label.setText(f"Submitted: {text}")
        self.input_box.clear()
        self.add_message("You", text)

        if self.on_submit is not None:
            self.status_label.setText("Thinking...")
            Thread(target=self._run_submit, args=(text,), daemon=True).start()

    def _run_submit(self, text):
        try:
            response = self.on_submit(text)
        except Exception as exc:
            response = f"Error: {exc}"

        self._chat_queue.put(response)

    def add_message(self, sender, text):
        self.chat_log.appendPlainText(f"{sender}: {text}")
        self.chat_log.verticalScrollBar().setValue(
            self.chat_log.verticalScrollBar().maximum()
        )
        self.app.processEvents()

    def set_mask_options(self, labels):
        labels = sorted({label.lower() for label in labels if label})

        if labels == sorted(self.mask_vars):
            return

        old_labels = set(self.mask_vars)
        selected = self.get_selected_masks()
        if not self._mask_options_initialized:
            selected = set(labels)
        else:
            selected |= set(labels) - old_labels

        self.mask_vars = {}
        self.mask_menu.clear()

        for label in labels:
            action = QAction(label, self.mask_menu)
            action.setCheckable(True)
            action.setChecked(label in selected)
            action.triggered.connect(lambda checked=False: self._handle_mask_change())
            self.mask_vars[label] = action
            self.mask_menu.addAction(action)

        self._mask_options_initialized = True
        self._handle_mask_change()

    def get_selected_masks(self):
        return {
            label
            for label, action in self.mask_vars.items()
            if action.isChecked()
        }

    def _handle_mask_change(self):
        selected = self.get_selected_masks()
        display = ", ".join(sorted(selected)) or "None"
        self.mask_menu_button.setText(f"Observed masks: {display}")

        if self.on_mask_change is not None:
            self.on_mask_change(selected)

    def _show_image(self, widget, frame):
        if frame is None:
            return

        widget.set_frame(frame)

    def show_frame(self, frame):
        self._show_image(self.image_label, frame)

    def show_graph(self, frame):
        self._show_image(self.graph_label, frame)

    def start_ui_recording(self, output_path, fps=30):
        self._ui_recording_path = output_path
        self._ui_recording_fps = fps

    def start_graph_recording(self, output_path, fps=30):
        self._graph_recording_path = output_path
        self._graph_recording_fps = fps

    def _capture_window(self, window):
        pixmap = window.grab()
        image = pixmap.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
        width = image.width()
        height = image.height()

        ptr = image.bits()
        if hasattr(ptr, "setsize"):
            ptr.setsize(image.sizeInBytes())
        rgba = np.frombuffer(
            ptr,
            np.uint8,
            count=image.sizeInBytes(),
        ).reshape((height, width, 4)).copy()
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)

    def _record_window_frame(self, window, recorder_attr, path_attr, fps_attr):
        output_path = getattr(self, path_attr)
        if output_path is None:
            return

        frame = self._capture_window(window)
        if frame is None:
            return

        height, width = frame.shape[:2]

        recorder = getattr(self, recorder_attr)
        if recorder is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            recorder = cv2.VideoWriter(
                str(output_path),
                fourcc,
                getattr(self, fps_attr),
                (width, height),
            )
            setattr(self, recorder_attr, recorder)

        recorder.write(frame)

    def _record_ui_frame(self):
        self._record_window_frame(
            self.root,
            "_ui_recorder",
            "_ui_recording_path",
            "_ui_recording_fps",
        )

    def _record_graph_frame(self):
        self._record_window_frame(
            self.graph_window,
            "_graph_recorder",
            "_graph_recording_path",
            "_graph_recording_fps",
        )

    def update(self):
        self.app.processEvents()

    def run_frame_loop(self, on_frame, delay_ms=30):
        self._running = True

        def worker():
            while self._running:
                result = on_frame()
                labels = None
                if isinstance(result, tuple) and len(result) == 3:
                    frame, graph_frame, labels = result
                elif isinstance(result, tuple):
                    frame, graph_frame = result
                else:
                    frame = result
                    graph_frame = None

                if frame is None:
                    self._running = False
                    break

                if self._frame_queue.full():
                    try:
                        self._frame_queue.get_nowait()
                    except Empty:
                        pass

                self._frame_queue.put((frame, graph_frame, labels))

        self._worker = Thread(target=worker, daemon=True)
        self._worker.start()
        self._poll_timer.start(delay_ms)
        self.run()

    def _poll_frame(self):
        self._poll_chat()

        try:
            frame, graph_frame, labels = self._frame_queue.get_nowait()
        except Empty:
            frame = None
            graph_frame = None
            labels = None

        if frame is not None:
            self.show_frame(frame)
        if graph_frame is not None:
            self.show_graph(graph_frame)
            self._record_graph_frame()
        if labels is not None:
            self.set_mask_options(labels)
        if frame is not None:
            self._record_ui_frame()

        if not self._running:
            self._poll_timer.stop()
            self.app.quit()

    def _poll_chat(self):
        while True:
            try:
                response = self._chat_queue.get_nowait()
            except Empty:
                break

            if response:
                self.add_message("System", response)
            self.status_label.setText("Ready")

    def run(self):
        self.app.exec()

    def stop(self):
        self._running = False
        self.app.quit()

    def close(self):
        self._running = False
        self._closing = True
        self._poll_timer.stop()
        if self._ui_recorder is not None:
            self._ui_recorder.release()
            self._ui_recorder = None
        if self._graph_recorder is not None:
            self._graph_recorder.release()
            self._graph_recorder = None
        self.graph_window.close()
        self.root.close()
        self.app.quit()
