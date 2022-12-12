import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms


# switching to cpu if possible
device = 'cpu'
if torch.cuda.device_count() > 0 and torch.cuda.is_available():
    print("Cuda installed! Running on GPU!")
    device = 'cuda'
else:
    print("No GPU available!")


class ConvLSTMCell(nn.Module):
    """
    The class called ConvLSTMCell is initialised with custumisable parameters,
    this is used to contruct the convolutional cell in the conv-LSTM
    """

    def __init__(self, in_channels, out_channels,
                 kernel_size, padding, activation, image_size):
        """
        Set up the fileter-parameters and constants for the convolution

        Parameters
        ----------
        in_channels : int
            Number of channels of input data

        out_channels : int
            Number of channels in the output data, thus number of kernelsc
            in convolution

        kernel_size : int
            size of kernel in convolution

        padding : int
            number of lines of padding around area spanned by kernel

        activation : pytorch function
            Function used to activate layer

        image_size : pytorch.tensor
            size of input image

        Examples
        --------
        >>> m = ConvLSTMCell(1, 3, 5, 2, torch.sigmoid, (360,360))
        >>> print(m)
        ConvLSTMCell(
          (conv): Conv2d(4, 12, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        )
        """  # noqa

        super(ConvLSTMCell, self).__init__()

        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = torch.relu

        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        self.conv = nn.Conv2d(
            in_channels=in_channels + out_channels,
            out_channels=4 * out_channels,
            kernel_size=kernel_size,
            padding=padding)

        # Initialize weights for Hadamard Products
        self.W_ci = nn.Parameter(torch.Tensor(out_channels, *image_size))
        self.W_co = nn.Parameter(torch.Tensor(out_channels, *image_size))
        self.W_cf = nn.Parameter(torch.Tensor(out_channels, *image_size))

    def forward(self, X, H_prev, C_prev):
        """
        forward pass through the cell

        Parameters
        ----------
        X : pytroch.tensor

        H_prev : pytorch.tensor

        C_prev : pytorch.tensor

        Returns
        -------
        C : pytorch.tensor
            current state

        H : pytorch.tensor
            Hidden state

        Examples
        --------
        TODO
        """

        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        conv_output = self.conv(torch.cat([X, H_prev], dim=1))

        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        i_conv, f_conv, C_conv, o_conv = torch.chunk(conv_output,
                                                     chunks=4, dim=1)

        input_gate = torch.sigmoid(i_conv + self.W_ci * C_prev)
        forget_gate = torch.sigmoid(f_conv + self.W_cf * C_prev)

        # Current Cell output
        C = forget_gate*C_prev + input_gate * self.activation(C_conv)

        output_gate = torch.sigmoid(o_conv + self.W_co * C)

        # Current Hidden State
        H = output_gate * self.activation(C)

        return H, C


class ConvLSTM(nn.Module):
    """
    The class called ConvLSTM is initialised with custumisable parameters,
    this is used to contruct the convolutional LSTM model network
    """

    def __init__(self, in_channels, out_channels,
                 kernel_size, padding, activation, image_size):
        """
        Set up the parameters and constants for the model

        Parameters
        ----------
        in_channels : int
            Number of channels of input data

        out_channels : int
            Number of channels in the output data, thus number of kernels
            in convolution

        kernel_size : int
            size of kernel in convolution

        padding : int
            number of lines of padding around area spanned by kernel

        activation : pytorch function
            Function used to activate layer

        image_size : pytorch.tensor
            size of input image

        Examples
        --------
        >>> m = ConvLSTMCell(1, 3, 5, 2, torch.sigmoid, (360,360))
        >>> print(m)
        ConvLSTMCell(
          (conv): Conv2d(4, 12, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        )
        """  # noqa

        super(ConvLSTM, self).__init__()

        self.out_channels = out_channels

        # We will unroll this over time steps
        self.convLSTMcell = ConvLSTMCell(in_channels, out_channels,
                                         kernel_size, padding,
                                         activation, image_size)

    def forward(self, X):
        """
        Function to conduct a single forward pass on Convolutional LSTM Network

        Parameters
        ----------
        X : pytroch.tensor

        Returns
        -------
        output : pytorch.tensor
            current state

        Examples
        --------
        TODO
        """
        # X is a frame sequence (batch_size,
        # num_channels, seq_len, height, width)

        # Get the dimensions
        batch_size, _, seq_len, height, width = X.size()

        # Initialize output
        output = torch.zeros(batch_size, self.out_channels, seq_len,
                             height, width, device=device)
        # Initialize Hidden State
        H = torch.zeros(batch_size, self.out_channels,
                        height, width, device=device)
        # Initialize Cell Input
        C = torch.zeros(batch_size, self.out_channels,
                        height, width, device=device)

        # Unroll over time steps
        for time_step in range(seq_len):

            H, C = self.convLSTMcell(X[:, :, time_step], H, C)

            output[:, :, time_step] = H

        return output


class Seq2Seq(nn.Module):
    """
    The class called Seq2Seq is initialised with custumisable parameters,
    this is used to contruct the convolutional LSTM model network
    with multiple layers
    """

    def __init__(self, num_channels, num_kernels, kernel_size, padding,
                 activation, image_size, num_layers):
        """
        Set up the parameters and constants for the model

        Parameters
        ----------
        num_channels : int
            Number of channels of input data

        num_kernels : int
            number of kernels in convolution

        kernel_size : int
            size of kernel in convolution

        padding : int
            number of lines of padding around area spanned by kernel

        activation : pytorch function
            Function used to activate layer

        image_size : pytorch.tensor
            size of input image

        num_layers : int
            number of layers in network

        Examples
        --------
        >>> m = Seq2Seq(3, 3, 5, 2, torch.sigmoid, (1,360,360), 3)
        >>> print(m)
        Seq2Seq(
          (sequential): Sequential(
            (convlstm1): ConvLSTM(
              (convLSTMcell): ConvLSTMCell(
                (conv): Conv2d(6, 12, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
              )
            )
            (batchnorm1): BatchNorm3d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (convlstm2): ConvLSTM(
              (convLSTMcell): ConvLSTMCell(
                (conv): Conv2d(6, 12, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
              )
            )
            (batchnorm2): BatchNorm3d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (convlstm3): ConvLSTM(
              (convLSTMcell): ConvLSTMCell(
                (conv): Conv2d(6, 12, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
              )
            )
            (batchnorm3): BatchNorm3d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (conv): Conv2d(3, 3, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        )

        """  # noqa

        super(Seq2Seq, self).__init__()

        self.sequential = nn.Sequential()

        # Add First layer (Different in_channels than the rest)
        self.sequential.add_module(
            "convlstm1", ConvLSTM(
                in_channels=num_channels, out_channels=num_kernels,
                kernel_size=kernel_size, padding=padding,
                activation=activation, image_size=image_size)
        )

        self.sequential.add_module(
            "batchnorm1", nn.BatchNorm3d(num_features=num_kernels)
        )

        # Add rest of the layers
        for idx in range(2, num_layers+1):

            self.sequential.add_module(
                f"convlstm{idx}", ConvLSTM(
                    in_channels=num_kernels, out_channels=num_kernels,
                    kernel_size=kernel_size, padding=padding,
                    activation=activation, image_size=image_size)
                )

            self.sequential.add_module(
                f"batchnorm{idx}", nn.BatchNorm3d(num_features=num_kernels)
                )

        # Add Convolutional Layer to predict output frame
        self.conv = nn.Conv2d(
            in_channels=num_kernels, out_channels=num_channels,
            kernel_size=kernel_size, padding=padding)

    def forward(self, X):
        """
        Function to conduct a single forward pass on Convolutional LSTM Network

        Parameters
        ----------
        X : pytroch.tensor

        Returns
        -------
        output : pytorch.tensor
            current state

        Examples
        --------
        TODO
        """
        # Forward propagation through all the layers
        output = self.sequential(X)

        # Return only the last output frame
        output = self.conv(output[:, :, -1])
        return nn.Sigmoid()(output)


def train_conv_lstm(model, optimizer, criterion, train_loader):
    """
    Run training sequence for the convolution LSTM Network

    Parameters
    ----------
    model : pytroch.model

    optimizer: pytorch function
        optimiser chosen to optimise parameters

    criterion: pytorch function
        function for calculating loss

    train_loader: pytorch.dataloader
        batch of data to used for training

    Returns
    -------
    train_loss : float
        loss at end of training

    Examples
    --------
    TODO
    """
    train_loss = 0
    model.train()

    for batch_num, (input, target) in enumerate(train_loader, 1):

        input, target = input.to(device), target.to(device)
        output = model(input)

        loss = criterion(output.flatten(), target.flatten())
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.item()

    train_loss /= len(train_loader.dataset)

    return train_loss


def validate_conv_lstm(model, val_loader, criterion):
    """
    Run validation sequence for the convolution LSTM Network

    Parameters
    ----------
    model : pytroch.model

    criterion: pytorch function
        function for calculating loss

    val_loader: pytorch.dataloader
        batch of data to used for validation

    Returns
    -------
    val_loss : float
        loss at end of validation

    Examples
    --------
    TODO
    """

    val_loss = 0
    model.eval()

    with torch.no_grad():
        for input, target in val_loader:
            input, target = input.to(device), target.to(device)
            output = model(input)

            loss = criterion(output.flatten(), target.flatten())
            val_loss += loss.item()

    val_loss /= len(val_loader.dataset)

    return val_loss


def eval_images(model, test_loader, test_num=5):
    """
    Function generating predicted images and compare with the real ones

    Parameters
    ----------
    model : pytroch.model

    test_num: int, optional
        number of images used for testing

    test_loader: pytorch.dataloader
        batch of data to used for test

    Returns
    -------
    output : pytorch.tensor, image
        predicted image

    target : pytorch.tensor
        class of each image, in our case time interval

    Examples
    --------
    TODO
    """

    model.eval()

    with torch.no_grad():
        fig, ax = plt.subplots(2, test_num, figsize=(20, 8))

    for i in range(test_num):
        input, target = next(iter(test_loader))
        input, target = input.to(device), target.to(device)
        output = model(input)
        ax[0, i].imshow(transforms.ToPILImage()(output[0]))
        ax[1, i].imshow(transforms.ToPILImage()(target[0]))
    return output, target
