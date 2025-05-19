from ai_edge_litert.interpreter import Interpreter

import tensorflow as tf
import numpy as np
import zipfile

model_path = "model.tflite"
interpreter = Interpreter(model_path)

input_details = interpreter.get_input_details()
waveform_input_index = input_details[0]["index"]
output_details = interpreter.get_output_details()
scores_output_index = output_details[0]["index"]

# Input: 0.975 seconds of silence as mono 16 kHz waveform samples.
waveform = np.zeros(int(round(0.975 * 16000)), dtype=np.float32)
print(waveform.shape)  # Should print (15600,)

interpreter.resize_tensor_input(waveform_input_index, [waveform.size], strict=True)
interpreter.allocate_tensors()
interpreter.set_tensor(waveform_input_index, waveform)
interpreter.invoke()
scores = interpreter.get_tensor(scores_output_index)
print(scores.shape)  # Should print (1, 521)

top_class_index = scores.argmax()
labels_file = zipfile.ZipFile(model_path).open("label.txt")
labels = [l.decode("utf-8").strip() for l in labels_file.readlines()]
print(len(labels))  # Should print 521
print(labels[top_class_index])  # Should print 'Silence'.
