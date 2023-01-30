import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Dense
from keras.utils.vis_utils import plot_model
from tensorflow.python.ops.gen_nn_ops import LeakyRelu
import datetime


def pointsIncircum(r, n=100):
    return [(math.cos(2 * math.pi / n * x) * r, math.sin(2 * math.pi / n * x) * r) for x in range(0, n + 1)]


circle = np.array(pointsIncircum(2, 1000))


# print(circle)
# fig = plt.figure()
# plt.scatter(circle[:, 0], circle[:, 1])
# plt.show()


def generator(latent_dim, n_outputs):
    model = Sequential(name="Generator")
    model.add(Dense(32, activation='relu', kernel_initializer='he_uniform', input_dim=latent_dim,
                    name='Generator-Hidden-Layer-1'))
    model.add(Dense(16, activation='relu', kernel_initializer='he_uniform', name='Generator-Hidden-Layer-2'))
    # model.add(LeakyRelu(alpha=0.5))
    model.add(Dense(n_outputs, activation='tanh', name='Generator-Output-Layer'))
    return model


def discriminator(n_inputs):
    model = Sequential(name='Discriminator')
    model.add(Dense(32, activation='relu', kernel_initializer='he_uniform', input_dim=n_inputs,
                    name='Discriminator-Hidden-Layer-1'))
    model.add(Dense(16, activation='relu', kernel_initializer='he_uniform', name='Discriminator-Hidden-Layer-2'))
    model.add(Dense(1, activation='sigmoid', name='Discriminator-Output-Layer'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def defGan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential(name='GAN')
    model.add(generator)
    model.add(discriminator)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


def real_samples(n, dataset):
    X = dataset.iloc[np.random.choice(dataset.shape[0], n, replace=True), :]
    y = np.ones((n, 1))
    return X, y


def latent_points(latent_dim, n):
    latent_input = np.random.randn(latent_dim * n)
    latent_input = latent_input.reshape(n, latent_dim)
    return latent_input


def fake_samples(generator, latent_dim, n):
    latent_output = latent_points(latent_dim, n)
    X = generator.predict(latent_output)
    y = np.zeros((n, 1))
    return X, y


def performance_summary(epoch, generator, discriminator, latent_dim, dataset, n=100):
    x_real, y_real = real_samples(n, dataset=dataset)
    _, real_accuracy = discriminator.evaluate(x_real, y_real, verbose=1)
    x_fake, y_fake = fake_samples(generator, latent_dim, n)
    _, fake_accuracy = discriminator.evaluate(x_fake, y_fake, verbose=1)
    print("Epoch number:", epoch)
    print("Discriminator Accuracy on real points", real_accuracy)
    print("Discriminator Accuracy on fake points", fake_accuracy)
    time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if 0.35 <= real_accuracy <= 0.65 and 0.35 <= fake_accuracy <= 0.65:
        x_fake = pd.DataFrame(x_fake)
        # if real_accuracy >= 0.85 and fake_accuracy >= 0.85:
        #     x_fake.to_csv(f"fakeMake{time}.csv", index=False)
        x_fake.to_csv(f"fakeMake{time}.csv", index=False)
        print(x_fake)

    # plt.figure()
    # plt.scatter(x_real.iloc[:, 0], x_real.iloc[:, 1], s=5)
    # plt.scatter(x_fake.iloc[:, 0], x_fake.iloc[:, 1], s=5)
    # plt.show()


def train(g_model, d_model, gan_model, latent_dim, dataset, n_epochs=100001, n_batch=256, n_eval=1000):
    half_batch = int(n_batch / 2)
    for i in range(n_epochs):
        x_real, y_real = real_samples(half_batch, dataset=dataset)
        x_fake, y_fake = fake_samples(g_model, latent_dim, half_batch)
        # print("fakeShape:", x_fake.shape)
        d_model.train_on_batch(x_real, y_real)
        d_model.train_on_batch(x_fake, y_fake)

        x_gan = latent_points(latent_dim, n_batch)
        y_gan = np.ones((n_batch, 1))
        gan_model.train_on_batch(x_gan, y_gan)
        if (i % n_eval == 0):
            performance_summary(i, g_model, d_model, latent_dim, dataset=dataset, n=284315)
            # print(x_fake, y_fake)


def GAN(dataset):
    latent_dim = 3
    n_dim = dataset.shape[1]
    gen_model = generator(latent_dim, n_outputs=n_dim)
    gen_model.summary()
    plot_model(gen_model, show_shapes=True, show_layer_names=True, dpi=400)
    dis_model = discriminator(n_inputs=n_dim)
    dis_model.summary()
    plot_model(dis_model, show_shapes=True, show_layer_names=True, dpi=400)
    gan_model = defGan(gen_model, dis_model)
    gan_model.summary()
    plot_model(gan_model, show_shapes=True, show_layer_names=True, dpi=400)
    train(gen_model, dis_model, gan_model, latent_dim, dataset=dataset)

# GAN(circle)
