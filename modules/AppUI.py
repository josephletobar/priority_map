import tkinter as tk
from tkinter.scrolledtext import ScrolledText
import ctypes
from ctypes import wintypes
from queue import Empty, Queue
from threading import Thread

import cv2
import numpy as np


user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32

PW_RENDERFULLCONTENT = 0x00000002
SRCCOPY = 0x00CC0020


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [
        ("bmiHeader", BITMAPINFOHEADER),
        ("bmiColors", wintypes.DWORD * 3),
    ]


class AppUI:
    def __init__(
        self,
        title="Drone Heatmap",
        on_submit=None,
        on_mask_change=None,
        max_display_size=(1400, 900)
    ):
        self.on_submit = on_submit
        self.on_mask_change = on_mask_change
        self.max_display_size = max_display_size

        self.root = tk.Tk()
        self.root.title(title)

        self.graph_window = tk.Toplevel(self.root)
        self.graph_window.title("Graph")
        self.graph_label = tk.Label(self.graph_window, bg="black")
        self.graph_label.pack(fill=tk.BOTH, expand=True)

        self.image_label = tk.Label(self.root, bg="black")
        self.image_label.pack(fill=tk.BOTH, expand=True)

        self.chat_log = ScrolledText(self.root, height=8, state=tk.DISABLED)
        self.chat_log.pack(fill=tk.BOTH, expand=False)

        self.status_var = tk.StringVar(value="Ready")
        self.status_label = tk.Label(
            self.root,
            textvariable=self.status_var,
            anchor="w"
        )
        self.status_label.pack(fill=tk.X)

        self.mask_vars = {}
        self._mask_options_initialized = False
        self.mask_menu_button = tk.Menubutton(
            self.root,
            text="Observed masks",
            relief=tk.RAISED
        )
        self.mask_menu = tk.Menu(self.mask_menu_button, tearoff=False)
        self.mask_menu_button.configure(menu=self.mask_menu)
        self.mask_menu_button.pack(fill=tk.X)

        input_frame = tk.Frame(self.root)
        input_frame.pack(fill=tk.X)

        self.input_var = tk.StringVar()
        self.input_box = tk.Entry(input_frame, textvariable=self.input_var)
        self.input_box.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.input_box.bind("<Return>", self._handle_submit)

        self.send_button = tk.Button(
            input_frame,
            text="Send",
            command=self._handle_submit
        )
        self.send_button.pack(side=tk.RIGHT)

        self._photo = None
        self._graph_photo = None
        self._running = False
        self._frame_queue = Queue(maxsize=1)
        self._chat_queue = Queue()
        self._worker = None
        self._ui_recorder = None
        self._ui_recording_path = None
        self._graph_recorder = None
        self._graph_recording_path = None
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _handle_submit(self, event=None):
        text = self.input_var.get().strip()
        if not text:
            return

        self.status_var.set(f"Submitted: {text}")
        self.input_var.set("")
        self.add_message("You", text)

        if self.on_submit is not None:
            self.status_var.set("Thinking...")
            Thread(
                target=self._run_submit,
                args=(text,),
                daemon=True,
            ).start()

    def _run_submit(self, text):
        try:
            response = self.on_submit(text)
        except Exception as exc:
            response = f"Error: {exc}"

        self._chat_queue.put(response)

    def add_message(self, sender, text):
        self.chat_log.configure(state=tk.NORMAL)
        self.chat_log.insert(tk.END, f"{sender}: {text}\n")
        self.chat_log.see(tk.END)
        self.chat_log.configure(state=tk.DISABLED)
        self.root.update_idletasks()

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
        self.mask_menu.delete(0, tk.END)

        for label in labels:
            var = tk.BooleanVar(value=label in selected)
            self.mask_vars[label] = var
            self.mask_menu.add_checkbutton(
                label=label,
                variable=var,
                command=self._handle_mask_change,
            )

        self._mask_options_initialized = True
        self._handle_mask_change()

    def get_selected_masks(self):
        return {
            label
            for label, var in self.mask_vars.items()
            if var.get()
        }

    def _handle_mask_change(self):
        selected = self.get_selected_masks()
        display = ", ".join(sorted(selected)) or "None"
        self.mask_menu_button.configure(text=f"Observed masks: {display}")

        if self.on_mask_change is not None:
            self.on_mask_change(selected)

    def _fit_display(self, frame):
        max_width, max_height = self.max_display_size
        height, width = frame.shape[:2]
        scale = min(max_width / width, max_height / height, 1.0)

        if scale >= 1.0:
            return frame

        return cv2.resize(
            frame,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )

    def _show_image(self, label, photo_attr, frame):
        if frame is None:
            return

        frame = self._fit_display(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = frame.shape[:2]
        data = b"P6\n%d %d\n255\n" % (width, height) + frame.tobytes()

        photo = tk.PhotoImage(master=self.root, data=data, format="PPM")
        setattr(self, photo_attr, photo)
        label.configure(image=photo)

    def show_frame(self, frame):
        self._show_image(self.image_label, "_photo", frame)

    def show_graph(self, frame):
        self._show_image(self.graph_label, "_graph_photo", frame)

    def start_ui_recording(self, output_path, fps=30):
        self._ui_recording_path = output_path
        self._ui_recording_fps = fps

    def start_graph_recording(self, output_path, fps=30):
        self._graph_recording_path = output_path
        self._graph_recording_fps = fps

    def _capture_window(self, window):
        hwnd = window.winfo_id()
        rect = RECT()

        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return None

        width = rect.right - rect.left
        height = rect.bottom - rect.top
        if width <= 1 or height <= 1:
            return None

        window_dc = user32.GetWindowDC(hwnd)
        memory_dc = gdi32.CreateCompatibleDC(window_dc)
        bitmap = gdi32.CreateCompatibleBitmap(window_dc, width, height)
        old_bitmap = gdi32.SelectObject(memory_dc, bitmap)

        try:
            ok = user32.PrintWindow(hwnd, memory_dc, PW_RENDERFULLCONTENT)
            if not ok:
                gdi32.BitBlt(memory_dc, 0, 0, width, height, window_dc, 0, 0, SRCCOPY)

            bitmap_info = BITMAPINFO()
            bitmap_info.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
            bitmap_info.bmiHeader.biWidth = width
            bitmap_info.bmiHeader.biHeight = -height
            bitmap_info.bmiHeader.biPlanes = 1
            bitmap_info.bmiHeader.biBitCount = 32
            bitmap_info.bmiHeader.biCompression = 0

            buffer = np.empty((height, width, 4), dtype=np.uint8)
            gdi32.GetDIBits(
                memory_dc,
                bitmap,
                0,
                height,
                buffer.ctypes.data_as(ctypes.c_void_p),
                ctypes.byref(bitmap_info),
                0,
            )

            return cv2.cvtColor(buffer, cv2.COLOR_BGRA2BGR)

        finally:
            gdi32.SelectObject(memory_dc, old_bitmap)
            gdi32.DeleteObject(bitmap)
            gdi32.DeleteDC(memory_dc)
            user32.ReleaseDC(hwnd, window_dc)

    def _record_window_frame(self, window, recorder_attr, path_attr, fps_attr):
        output_path = getattr(self, path_attr)
        if output_path is None:
            return

        window.update_idletasks()
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
        self.root.update_idletasks()
        self.root.update()

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

        def poll_frame():
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

            if self._running:
                self.root.after(delay_ms, poll_frame)
            else:
                self.root.quit()

        self._worker = Thread(target=worker, daemon=True)
        self._worker.start()
        self.root.after(delay_ms, poll_frame)
        self.run()

    def _poll_chat(self):
        while True:
            try:
                response = self._chat_queue.get_nowait()
            except Empty:
                break

            if response:
                self.add_message("System", response)
            self.status_var.set("Ready")

    def run(self):
        self.root.mainloop()

    def stop(self):
        self._running = False
        self.root.quit()

    def close(self):
        self._running = False
        if self._ui_recorder is not None:
            self._ui_recorder.release()
            self._ui_recorder = None
        if self._graph_recorder is not None:
            self._graph_recorder.release()
            self._graph_recorder = None
        if self.graph_window.winfo_exists():
            self.graph_window.destroy()
        if self.root.winfo_exists():
            self.root.destroy()
