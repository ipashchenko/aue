from keras.layers import Input, Dense
from keras.models import Model
from sklearn.base import TransformerMixin, BaseEstimator
import matplotlib.pyplot as plt


class AETransform(TransformerMixin, BaseEstimator):
    def __init__(self, code_dim=15, dims=None, loss=None):
        """
        Autoencoder transformer. Note that because sigmoid activation used at
        the final layer of the decoder the data must be in (0, 1) range for
        "binary_crossentropy" loss to get positive values.

        :param code_dim:
            Dimension of code.
        :param dims:
            Iterable of dimensions for each of layers (except coding). E.g.
            ``[90, 60, 30]`` - means that there will be 3 layers with 90, 60 and
            30 number of units and coding layes, specified by ``code_dim``.
        """
        super(AETransform, self).__init__()
        self.encoder = None
        self.decoder = None
        self.autoencoder = None
        self.code_dim = code_dim
        self.dims = dims
        self.loss = loss
        self.history = None

    def fit(self, X, y=None, **fit_params):
        ncol = X.shape[1]
        input_dim = Input(shape=(ncol,))
        encoding_dim = Input(shape=(self.code_dim,))

        x = Dense(self.dims[0], activation='relu')(input_dim)
        for dim in self.dims[1:]:
            x = Dense(dim, activation='relu')(x)
        encoded = Dense(self.code_dim, activation='linear')(x)

        x = Dense(self.dims[-1], activation='relu')(encoded)
        for dim in self.dims[-2::-1]:
            x = Dense(dim, activation='relu')(x)
        decoded = Dense(ncol, activation='sigmoid')(x)

        encoder = Model(inputs=input_dim, outputs=encoded, name='encoder')
        self.encoder = encoder

        self.autoencoder = Model(inputs=input_dim, outputs=decoded)
        # Possible way:
        # self.autoencoder = Model(inputs=input_dim,
        #                          outputs=decoder(encoder(input_dim)))

        deco = self.autoencoder.layers[-3](encoding_dim)
        deco = self.autoencoder.layers[-2](deco)
        deco = self.autoencoder.layers[-1](deco)
        decoder = Model(inputs=encoding_dim, outputs=deco, name='decoder')
        self.decoder = decoder

        self.autoencoder.compile(optimizer='adam', loss=self.loss)
        history = self.autoencoder.fit(X, X, shuffle=True, verbose=2,
                                       **fit_params).history
        self.history = history

        return self

    def summary(self):
        if self.autoencoder is not None:
            return self.autoencoder.summary()

    def transform(self, X, **transform_params):
        return self.autoencoder.predict(X)

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, **fit_params).transform(X)

    def plot_history(self, save_file=None):
        fig, axes = plt.subplots(1, 1)
        axes.plot(self.history['loss'])
        axes.plot(self.history['val_loss'])
        axes.set_title('model loss')
        axes.set_ylabel('loss')
        axes.set_xlabel('epoch')
        axes.legend(['train', 'test'], loc='upper right')
        fig.tight_layout()
        if save_file:
            fig.savefig(save_file, dpi=300)
