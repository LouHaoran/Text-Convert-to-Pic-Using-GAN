import gan

if __name__ == '__main__':
    gan.train_gan(examples=2, epochs=10, batch_size=4, smaller=0.00008)  # For now, use small fraction of set