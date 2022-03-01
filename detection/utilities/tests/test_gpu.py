# import os
# import logging
# logging.getLogger("tensorflow").disabled = True
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

print("_" * 65)
print("If you was setup CUDA and num GPU >0 it is success")

cpu_list = tf.config.experimental.list_physical_devices("CPU")
print("\nNum CPUs Available: ", len(cpu_list))
[print(f"{item.name}") for item in cpu_list]

gpu_list = tf.config.experimental.list_physical_devices("GPU")
print("\nNum GPUs Available: ", len(gpu_list))
[print(f"{item.name}") for item in gpu_list]

tpu_list = tf.config.experimental.list_physical_devices("TPU")
print("\nNum TPUs Available: ", len(tpu_list))
[print(f"{item.name}") for item in tpu_list]
print("_" * 65)
