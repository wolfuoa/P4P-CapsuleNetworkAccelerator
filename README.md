# P4P-CapsuleNetworkAccelerator

This repository provides a hardware-accelerated implementation of a Capsule Network (CapsNet), designed to enhance the performance of machine learning models, particularly in dynamic routing and classification tasks. The accelerator is optimized for use on FPGA platforms, leveraging hardware-specific optimization techniques to improve speed and energy efficiency.

Contact Information: nicholaswolf314@gmail.com

## Getting Started
### Prerequisites

Hardware: A Xilinx Zynq Ultrascale+ HMPSoC ZCU102.
    
Software: [Vitis, Vivado, Vitis HLS](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vitis.html)

License: Ask supervisor for Vitis license and installation instructions once Vivado is installed

### Installation

Clone the repository:

```bash
git clone https://github.com/wolfuoa/P4P-CapsuleNetworkAccelerator.git
cd P4P-CapsuleNetworkAccelerator
```

## C++ Capsule Network

The following section instructs the reader on how to run the C++ implementation of the Capsule Network.

### CMake
Before you begin, install [CMake](https://cmake.org/).

### Running the code
* Navigate to the `src` directory
```bash
cd src
```
* Follow the instructions within the enclosed ReadMe file

## Accelerated Capsule Network
The following section instructs the reader on how to run the Capsule Network accelerator on the ZCU102.

### Creating the DPU `.xmodel`
The `.xmodel` file is a graph of instructions to be run on the Xilinx DPU IP. In this research project, the DPU serves to accelerate the ReLU Convolution 1 and Primary Capsule layers of the Capsule Network.

* Install [Docker](https://www.docker.com/products/docker-desktop/)
* Perform a quick and simple test of your Docker installation by executing the following command. This command will download a test image from Docker Hub and run it in a container. When the container runs successfully, it prints a “Hello World” message and exits.
```bash
[Host] $ docker run hello-world
```
* Finally, verify that the version of Docker that you have installed meets the minimum Host System Requirements by running the following command
```bash
[Host] $ docker --version
```

* Clone Vitis-AI
```bash
[Host] $ git clone https://github.com/Xilinx/Vitis-AI.git
```
* Run Docker
```bash
[Host] $ sudo docker
```
* Activate TensorFlow2 Environment

> [!NOTE]
> New terminal. Don't close the Docker daemon

```bash
[Host] $ cd <Vitis-AI Installation Directory>
[Host] $ docker pull xilinx/vitis-ai-tensorflow2-cpu:latest
[Host] $ ./docker_run.sh xilinx/vitis-ai-tensorflow2-cpu:latest
[Docker] $ conda activate vitis-ai-tensorflow2
[Docker] $ pip install torch
```
* Copy files into workspace
> [!NOTE]
> New terminal.
```bash
[Host] $ cd <P4P-CapsuleNetworkAccelerator Installation Directory>
[Host] $ sudo cp -r deployment <Vitis-AI Installation Directory>
```
* Build the deconstructed Capsule Network
```bash
[Docker] $ cd deployment
[Docker] $ python3 create_partial_model.py
```
* Quantize the Capsule Network
```bash
[Docker] $ python3 quantizer.py
```
* Compile the `.xmodel`
```bash
[Docker] $ bash -x compile.sh
```
Completing the above steps will produce `compiled_model/partial_caps.xmodel`.


### Generating the SD-card image
The following section instructs the reader on how to generate the SD-card image containing the HLS kernel and DPU IP.
> [!WARNING]
> This will take 2-3 hours!

:pushpin: **Note:** This application can be run only on **Zynq UltraScale+ ZCU102**

* Download and unzip [MPSoC Common System](https://www.xilinx.com/member/forms/download/xef.html?filename=xilinx-zynqmp-common-v2022.1_04191534.tar.gz)
* Use [GitZip](https://kinolien.github.io/gitzip/) to download the [ZCU102 Base Platform](https://github.com/Xilinx/Vitis_Embedded_Platform_Source/tree/5ac2e444d985c069af8ed57e9f4eb9cbb911d075/Xilinx_Official_Platforms/xilinx_zcu102_base) and extract the files
* Download [PetaLinux](https://www.xilinx.com/member/forms/download/xef.html?filename=petalinux-v2024.1-05202009-installer.run)
* Download [XRT](https://www.xilinx.com/bin/public/openDownload?filename=xrt_202410.2.17.319_20.04-amd64-xrt.deb) (Ubuntu 20.04)
    * Probably installed in `/opt/xilinx/xrt`

* Build the Base Platform

Vitis and PetaLinux environment need to be setup before building the platform.

```bash
[Host] $ source <Vitis Installation Directory>/Vitis/2024.1/settings64.sh
[Host] $ source <PetaLinux Installation Directory>/settings.sh

[Host] $ cd <ZCU102 Base Platform Installation Directory>
[Host] $ make all
```
This should generate a `.xpfm` file.

* Build the image
> [!WARNING]
> This may take a while!

```bash
[Host] $ source <Vitis Installation Directory>/Vitis/2024.1/settings64.sh
[Host] $ source <XRT Installation Directory>/setup.sh
[Host] $ gunzip <MPSoC Common System>/xilinx-zynqmp-common-v2022.1/rootfs.tar.gz
[Host] $ export EDGE_COMMON_SW=<MPSoC Common System>/xilinx-zynqmp-common-v2022.1
[Host] $ export SDX_PLATFORM=<ZCU102 Base Platform Directory>/xilinx_zcu102_base_202210_1/xilinx_zcu102_base_202210_1.xpfm
[Host] $ export DEVICE=$SDX_PLATFORM
[Host] $ cd <P4P-CapsuleNetworkAccelerator>/accel/app/CapsuleNetwork/build_flow/DPUCZDX8G_zcu102
[Host] $ bash -x run.sh
```

> [!NOTE]
> - Generated SD card image will be here `<P4P-CapsuleNetworkAccelerator>/accel/app/CapsuleNetwork/build_flow/DPUCZDX8G_zcu102/binary_container_1/sd_card.img`.
> - The default setting of DPU is **B4096** with RAM_USAGE_LOW, CHANNEL_AUGMENTATION_ENABLE, DWCV_ENABLE, POOL_AVG_ENABLE, RELU_LEAKYRELU_RELU6, Softmax. Modify `<P4P-CapsuleNetworkAccelerator>accel/app/CapsuleNetwork/build_flow/DPUCZDX8G_zcu102/dpu_conf.vh` to change these.
> - Build runtime is ~1.5 hours.


### Running the Accelerator
The following section instructs the reader on how to run the accelerated Capsule Network.

* Flash the SD-Card with the `sd_card.img` using [Balena Etcher](https://etcher.balena.io/)

Once the SD-Card is set with `sd_card.img` system, next step is to install cross-compilation environment on the host system and the cross-compile the Capsule Network application.

  * Download the [sdk-2022.1.0.0.sh](https://www.xilinx.com/bin/public/openDownload?filename=sdk-2022.1.0.0.sh).

  * Install the cross-compilation system environment, follow the prompts to install. 

    **Please install it on your local host linux system, not in the docker system.**
```sh
[Host] $ ./sdk-2022.1.0.0.sh
```

Note that the `~/petalinux_sdk` path is recommended for the installation. Regardless of the path you choose for the installation, make sure the path has read-write permissions. 

Here we install it under `~/petalinux_sdk`.

 * When the installation is complete, follow the prompts and execute the following command.
```sh
[Host] $ source ~/petalinux_sdk/environment-setup-cortexa72-cortexa53-xilinx-linux
```

Note that if you close the current terminal, you need to re-execute the above instructions in the new terminal interface.

  * Download the [vitis_ai_2022.1-r2.5.0.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=vitis_ai_2022.1-r2.5.0.tar.gz) and install it to the petalinux system.
```sh
[Host] $ tar -xzvf vitis_ai_2022.1-r2.5.0.tar.gz -C ~/petalinux_sdk/sysroots/cortexa72-cortexa53-xilinx-linux
```

  * Cross compile `CapsuleNetwork`

```sh
[Host] $ cd  <P4P-CapsuleNetworkAccelerator>/accel/app/CapsuleNetwork
[Host] $ bash -x app_build.sh
```
If the compilation process does not report any error and the executable file `./bin/CapsuleNetwork.exe` is generated, then the host environment is installed correctly.

* Download  [Vitis AI Runtime 2.5.0](https://www.xilinx.com/bin/public/openDownload?filename=vitis-ai-runtime-2.5.0.tar.gz)
* Untar the runtime packet
```bash
[Host] $ tar -xzvf vitis-ai-runtime-2.5.0.tar.gz -C ./vitis-runtime
```

* Create `capsnet` in the BOOT partition `/run/media/mmcblk0p1/` of the SD-Card. Then copy the following contents to the `capsnet` directory of the BOOT partition of the SD-Card.
```sh
vitis-runtime
linux/model
linux/testing.sh
op_registration
bin   
linux/img
linux/setup.sh
```

> [!IMPORTANT]
> Switch ZCU102 into BOOT configuration 
> SW6 [4:1] = [OFF, OFF, OFF, ON]

* Connect USB-UART to board and open a serial terminal
    * Baud Rate: 115200
    * Data Bit: 8
    * Stop Bit: 1
    * No Parity
    * ID is most likely dev/ttyUSB0



```bash
[Host] $ sudo apt-get install -y putty
[Host] $ sudo putty
```

* Insert the SD card into the destination ZCU102 board and plugin the power. Connect serial port of the board to the host system. Wait for the Linux boot to complete. 

* Install the Vitis AI Runtime on the board. Execute the following command.

> [!NOTE]
> Only do this once

```bash
[Target] $ cd /run/media/mmcblk0p1/
[Target] $ cp -r vitis-ai-runtime-2.5.0/2022.1/aarch64/centos ~/
[Target] $ cd ~/centos
[Target] $ bash setup.sh
```

* Setup the application

```bash
[Target] $ cd /run/media/mmcblk0/capsnet
[Target] $ ./setup.sh
[Target] $ cd op_registration/cpp
[Target] $ ./op_registration.sh
```

* Run the application
```bash
[Target] $ cd /run/media/mmcblk0/capsnet
[Target] $ ./testing.sh
```

* Modify `testing.sh` using `vi` if you wish


### And that's all - Thank you for reading


