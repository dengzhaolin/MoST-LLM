import numpy as np
import h5py

train = []
test = []
val=[]
def get_embedding_train():
    train_embedding_path = "/nvme-data/dzl2023/MoSSL/module/fine_tune_merge/Embeddings/train/"

    for i in range(3869):
        file_path = f"{train_embedding_path}train_{i}.h5"
        with h5py.File(file_path, 'r') as hf:
            embedding = hf['embeddings'][:]  # 读取embedding


        # 在embedding的第一维添加一维
        #embedding = np.expand_dims(embedding, axis=0)  # 形状变为(1, 4, 1, 2048, 98)

        train.append(embedding)  # 将修改后的embedding添加到列表中
        print(f"Processed file {i}, train_length =", len(train))

    # 将列表中的所有embedding在第一维拼接
    train_embeddings = np.concatenate(train, axis=0)  # 形状为(3869, 4, 1, 2048, 98)

    # 保存到新的h5文件
    file_path = f"{train_embedding_path}train_embedding.h5"
    with h5py.File(file_path, 'w') as hf:
        hf.create_dataset('embeddings', data=train_embeddings)

    print("Ending")

def get_embedding_test():
    test_embedding_path = "/nvme-data/dzl2023/MoSSL/module/fine_tune_merge/Embeddings/test/"

    for i in range(240):
        file_path = f"{test_embedding_path}test_{i}.h5"
        with h5py.File(file_path, 'r') as hf:
            embedding = hf['embeddings'][:]  # 读取embedding

        # 在embedding的第一维添加一维
        #embedding = np.expand_dims(embedding, axis=0)  # 形状变为(1, 4, 1, 2048, 98)
        test.append(embedding)  # 将修改后的embedding添加到列表中
        print(f"Processed file {i}, train_length =", len(test))

    # 将列表中的所有embedding在第一维拼接
    test_embeddings = np.concatenate(test, axis=0)  # 形状为(3869, 4, 1, 2048, 98)

    # 保存到新的h5文件
    file_path = f"{test_embedding_path}test_embedding.h5"
    with h5py.File(file_path, 'w') as hf:
        hf.create_dataset('embeddings', data=test_embeddings)

    print("Ending")

def get_embedding_val():
    val_embedding_path = "/nvme-data/dzl2023/MoSSL/module/fine_tune_merge/Embeddings/val/"

    for i in range(240):
        file_path = f"{val_embedding_path}val_{i}.h5"
        with h5py.File(file_path, 'r') as hf:
            embedding = hf['embeddings'][:]  # 读取embedding

        # 在embedding的第一维添加一维
        #embedding = np.expand_dims(embedding, axis=0)  # 形状变为(1, 4, 1, 2048, 98)
        val.append(embedding)  # 将修改后的embedding添加到列表中
        print(f"Processed file {i}, train_length =", len(val))

    # 将列表中的所有embedding在第一维拼接
    val_embeddings = np.concatenate(val, axis=0)  # 形状为(3869, 4, 1, 2048, 98)

    # 保存到新的h5文件
    file_path = f"{val_embedding_path}val_embedding.h5"
    with h5py.File(file_path, 'w') as hf:
        hf.create_dataset('embeddings', data=val_embeddings)

    print("Ending")


