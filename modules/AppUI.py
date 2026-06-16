import tkinter as tk
from tkinter.scrolledtext import ScrolledText
from queue import Empty, Queue
from threading import Thread

import cv2


class AppUI:
    def __init__(self, title="Drone Heatmap", on_submit=None, max_display_size=(1400, 900)):
        self.on_submit = on_submit
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
        self._worker = None
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _handle_submit(self, event=None):
        text = self.input_var.get().strip()
        if not text:
            return

        self.status_var.set(f"Submitted: {text}")
        self.input_var.set("")
        self.add_message("You", text)

        if self.on_submit is not None:
            response = self.on_submit(text)
            if response:
                self.add_message("System", response)

    def add_message(self, sender, text):
        self.chat_log.configure(state=tk.NORMAL)
        self.chat_log.insert(tk.END, f"{sender}: {text}\n")
        self.chat_log.see(tk.END)
        self.chat_log.configure(state=tk.DISABLED)
        self.root.update_idletasks()

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

        ok, buffer = cv2.imencode(".png", frame)
        if not ok:
            return

        photo = tk.PhotoImage(master=self.root, data=buffer.tobytes())
        setattr(self, photo_attr, photo)
        label.configure(image=photo)

    def show_frame(self, frame):
        self._show_image(self.image_label, "_photo", frame)

    def show_graph(self, frame):
        self._show_image(self.graph_label, "_graph_photo", frame)

    def update(self):
        self.root.update_idletasks()
        self.root.update()

    def run_frame_loop(self, on_frame, delay_ms=30):
        self._running = True

        def worker():
            while self._running:
                result = on_frame()
                if isinstance(result, tuple):
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

                self._frame_queue.put((frame, graph_frame))

        def poll_frame():
            try:
                frame, graph_frame = self._frame_queue.get_nowait()
            except Empty:
                frame = None
                graph_frame = None

            if frame is not None:
                self.show_frame(frame)
            if graph_frame is not None:
                self.show_graph(graph_frame)

            if self._running:
                self.root.after(delay_ms, poll_frame)
            else:
                self.root.quit()

        self._worker = Thread(target=worker, daemon=True)
        self._worker.start()
        self.root.after(delay_ms, poll_frame)
        self.run()

    def run(self):
        self.root.mainloop()

    def stop(self):
        self._running = False
        self.root.quit()

    def close(self):
        self._running = False
        if self.graph_window.winfo_exists():
            self.graph_window.destroy()
        if self.root.winfo_exists():
            self.root.destroy()
