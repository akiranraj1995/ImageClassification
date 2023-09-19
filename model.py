import numpy as np
from kivy.utils import platform

if platform == 'android':
    from jnius import autoclass
    from android.permissions import request_permissions, Permission

    request_permissions([
        Permission.CAMERA,
        Permission.READ_EXTERNAL_STORAGE,
        Permission.WRITE_EXTERNAL_STORAGE])

    File = autoclass('java.io.File')
    Interpreter = autoclass('org.tensorflow.lite.Interpreter')
    InterpreterOptions = autoclass('org.tensorflow.lite.Interpreter$Options')
    TensorBuffer = autoclass(
        'org.tensorflow.lite.support.tensorbuffer.TensorBuffer')
    ByteBuffer = autoclass('java.nio.ByteBuffer')

    class TensorFlowModel():
        def load(self, model_filename, num_threads=None):
            model = File(model_filename)
            options = InterpreterOptions()
            if num_threads is not None:
                options.setNumThreads(num_threads)
            self.interpreter = Interpreter(model, options)
            self.allocate_tensors()

        def allocate_tensors(self):
            self.interpreter.allocateTensors()
            self.input_shape = self.interpreter.getInputTensor(0).shape()
            self.output_shape = self.interpreter.getOutputTensor(0).shape()
            self.output_type = self.interpreter.getOutputTensor(0).dataType()

        def get_input_shape(self):
            return self.input_shape

        def resize_input(self, shape):
            if self.input_shape != shape:
                self.interpreter.resizeInput(0, shape)
                self.allocate_tensors()

        def classify_image(self, image_data):
            input_tensor = ByteBuffer.wrap(image_data.tobytes())
            output_tensor = TensorBuffer.createFixedSize(self.output_shape,
                                                         self.output_type)
            self.interpreter.run(input_tensor, output_tensor.getBuffer().rewind())
            output = np.array(output_tensor.getFloatArray())
            class_index = np.argmax(output)
            classes = ['Apple', 'Banana', 'Grape', 'Mango', 'Strawberry']
            return classes[class_index]

else:
    import tensorflow as tf
    from PIL import Image

    class TensorFlowModel:
        def load(self, model_filename, num_threads=None):
            self.interpreter = tf.lite.Interpreter(model_filename,
                                                   num_threads=num_threads)
            self.interpreter.allocate_tensors()

        def resize_input(self, shape):
            if list(self.get_input_shape()) != shape:
                self.interpreter.resize_tensor_input(0, shape)
                self.interpreter.allocate_tensors()

        def get_input_shape(self):
            return self.interpreter.get_input_details()[0]['shape']

        def classify_image(self, image_data):
            input_details = self.interpreter.get_input_details()
            output_details = self.interpreter.get_output_details()

            # Preprocess the input image (assuming image_data is a NumPy array)
            input_data = np.expand_dims(image_data, axis=0)

            # Set the input tensor
            self.interpreter.set_tensor(input_details[0]['index'], input_data)

            # Run inference
            self.interpreter.invoke()

            # Get the output tensor and classify the image
            output_data = self.interpreter.get_tensor(output_details[0]['index'])
            class_index = np.argmax(output_data)
            classes = ['Apple', 'Banana', 'Grape', 'Mango', 'Strawberry']
            return classes[class_index]
