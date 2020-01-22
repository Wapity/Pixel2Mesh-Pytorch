
"""For training list"""
# with open("/Users/alex/Desktop/Pixel2Mesh/Pixel2Mesh-Pytorch-TUM/pytorch/data/training_data/train_list_old.txt", "r") as f:
#     content = f.read().splitlines()
#     with open("/Users/alex/Desktop/Pixel2Mesh/Pixel2Mesh-Pytorch-TUM/pytorch/data/training_data/trainer_list.txt", "w") as writer:
#         for path in content :
#             writer.write("d"+path[1:5]+"training_data/"+path[5:]+"\n")


"""For str list"""
# with open("/Users/alex/Desktop/Pixel2Mesh/Pixel2Mesh-Pytorch-TUM/pytorch/data/training_data/trainer_list.txt", "r") as f:
#     content = f.read().splitlines()
#     with open("/Users/alex/Desktop/Pixel2Mesh/Pixel2Mesh-Pytorch-TUM/pytorch/data/training_data/train_list_str.txt", "w") as writer:
#         for path in content :
#             if path.endswith("00.dat"):
#                 writer.write(path + "," + path[:-3] + "png" + "," + path[:-6] + "03.png\n")
