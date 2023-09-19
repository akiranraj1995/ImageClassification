import os
import kivy
from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.popup import Popup
from kivy.utils import platform
from kivy.resources import resource_find
from kivy.uix.filechooser import FileChooser
from PIL import Image as PILImage
import numpy as np
from model import TensorFlowModel  # You need to import your model here

kivy.require("2.2.1")


class ImageClassifierApp(App):
    def build(self):
        self.title = "Fruits Image Classification App"

        # Load labels from APK assets
        labels_path = resource_find("labels.txt")  # Updated asset access
        self.labels = self.load_labels(labels_path)

        # Use a ScrollView to make the UI scrollable
        layout = GridLayout(cols=1, spacing=10, padding=10)
        scroll_view = ScrollView()

        # Input elements (folder input and select folder button)
        input_layout = GridLayout(cols=2, spacing=10, size_hint_y=None, height=50)

        self.folder_input = TextInput(hint_text="Enter folder path or tap 'Select Folder'")
        self.select_folder_button = Button(text="Select Folder")
        self.select_folder_button.bind(on_release=self.select_folder)

        # Ensure the input field expands horizontally
        self.folder_input.size_hint_x = 0.8

        input_layout.add_widget(self.folder_input)
        input_layout.add_widget(self.select_folder_button)

        # Output elements (result label)
        self.result_label = Label()

        # "Predict" button in a separate BoxLayout
        predict_layout = BoxLayout(orientation="horizontal", spacing=10, size_hint_y=None, height=50)
        self.predict_button = Button(text="Predict", background_color=(1, 0, 0, 1))
        self.predict_button.bind(on_release=self.predict_images_in_folder)

        predict_layout.add_widget(self.predict_button)

        # Add all the elements to the main layout
        layout.add_widget(input_layout)
        layout.add_widget(self.result_label)
        layout.add_widget(predict_layout)

        scroll_view.add_widget(layout)  # Add the layout to the ScrollView

        # Create an instance of TensorFlowModel and load the model
        self.model = TensorFlowModel()
        model_path = resource_find("model.tflite")  # Updated asset access
        self.model.load(model_path)

        return scroll_view  # Return the ScrollView as the root widget

    def load_labels(self, path):
        try:
            with open(path, "r") as file:
                return file.read().splitlines()
        except FileNotFoundError:
            return []

    def predict_images_in_folder(self, instance):
        folder_path = self.folder_input.text
        if folder_path and os.path.exists(folder_path):
            self.result_label.text = ""

            def predict_subfolder(subfolder_path):
                subfolder_total_count = 0
                subfolder_correct_count = 0

                for image_filename in os.listdir(subfolder_path):
                    if image_filename.lower().endswith((".jpg", ".jpeg", ".png")):
                        image_path = os.path.join(subfolder_path, image_filename)
                        subfolder_total_count += 1

                        image = self.load_and_preprocess_image(image_path, (320, 240))

                        predicted_label = self.model.classify_image(image)

                        true_label = os.path.basename(subfolder_path)
                        if true_label == predicted_label:
                            subfolder_correct_count += 1

                self.result_label.text += (
                    f"Subfolder: {os.path.basename(subfolder_path)}, "
                    f"Total Images: {subfolder_total_count}, "
                    f"Correct: {subfolder_correct_count}, "
                    f"Incorrect: {subfolder_total_count - subfolder_correct_count}\n"
                )

            try:
                for subfolder in os.listdir(folder_path):
                    subfolder_path = os.path.join(folder_path, subfolder)
                    if os.path.isdir(subfolder_path):
                        predict_subfolder(subfolder_path)

            except NotADirectoryError:
                self.result_label.text = "Invalid folder path. Please enter a valid directory."
                # Keep the button color red when the path is invalid
                self.predict_button.background_color = (1, 0, 0, 1)
                return

            # Change the button color to green when a valid path is entered
            self.predict_button.background_color = (0, 1, 0, 1)
        else:
            self.result_label.text = "Invalid folder path. Please enter a valid directory."
            # Keep the button color red when the path is invalid
            self.predict_button.background_color = (1, 0, 0, 1)

    def load_and_preprocess_image(self, image_path, target_size):
        image = PILImage.open(image_path).resize(target_size)
        image = np.asarray(image)
        image = (image.astype(np.float32) / 255.0)
        return image

    def select_folder(self, instance):
        if platform == "android":
            # Handle Android file selection differently if needed
            pass
        else:
            file_chooser = FileChooser()
            file_chooser.bind(on_submit=self.handle_folder_selection_desktop)  # Use on_submit event
            self.popup = Popup(title="Select a folder", content=file_chooser, size_hint=(None, None), size=(400, 400))
            self.popup.open()  # Open the popup

    def handle_folder_selection_desktop(self, instance):
        folder_path = instance.selection and instance.selection[0] or ''
        self.folder_input.text = folder_path
        self.popup.dismiss()  # Close the popup


if __name__ == "__main__":
    ImageClassifierApp().run()
