## Overview
This bsp source code is generated from published bsp [ZCU102 BSP](https://www.xilinx.com/member/forms/download/xef.html?filename=xilinx-zcu102-v2022.1-04191534.bsp). The main differences are listed below:
- This project contains DPU of 3 cores B4096 with RAM_USAGE_LOW, CHANNEL_AUGMENTATION_ENABLE, DWCV_ENABLE, POOL_AVG_ENABLE, RELU_LEAKYRELU_RELU6, Softmax
- This project contains pre-installed recipes of Vitis AI Runtime and Library v2.5
- This project contains pre-installed recipes of app/dpu_sw_optimize.tar.gz for auto resize ext4 partiton and QoS optimization
- This project contains pre-installed recipes of resnet50 examples
- This project supports saving of uboot environmnet variables on uboot command line
- This project enables auto-login with root for development mode, which is not recommanded in your own production.



### Target Information

|Items|version information|
|:----:|:------:|
|petalinux tool| 2022.1|
|vivado |2022.1|
|target board|ZCU102|
|DPU version| dpuczdx8g-4.0 |

### Software Configurations
The software configurations are based on [ZCU102 BSP](https://www.xilinx.com/member/forms/download/xef.html?filename=xilinx-zcu102-v2022.1-04191534.bsp). Here is the list of additional configurations.

| Configuration |Items|Details|
|----|-----|------|
|Additional Kernel Configurations| CONFIG_XILINX_DPU=y| Patch for DPU compatibale-with-dpuczdx8g-4.0 |
|Additional RootFS Configurations|dnf<br />tcl<br />e2fsprogs-resize2fs<br />parted<br />libdfx:disabled<br />libmali-xlnx<br />opencl-clhpp-dev<br />mali-backend-x11<br />libstdcPLUSPLUS<br />gdb<br />python3<br />valgrind<br />resieze-part<br />libdrm, libdrm-tests and libdrm-kms<br />packagegroup-petalinux-audio<br />packagegroup-petalinux-gstreamer<br />packagegroup-petalinux-matchbox<br />packagegroup-petalinux-opencv<br />packagegroup-petalinux-self-hosted<br />packagegroup-petalinux-v4lutils<br />packagegroup-petalinux-vitisai<br />packagegroup-petalinux-x11<br />imagefeature-package-management<br />auto-login | NOTE: auto-login with root for development mode, which is not recommanded in your own production |
|Additional recipes Included|dpu-sw-optimize<br />recipes-vitis-ai<br />resnet50| |



**NOTE**:
1. When building PetaLinux image from source code, the default build temp directory is set to ${PROOT}/build/tmp/. To update the build tmp directory:
- Modify CONFIG_TMP_DIR_LOCATION macro in `./project-spec/configs/config file`


2. The DPU require continuous physical memory, which can be implemented by CMA. In platform source, the CMA is set to 1536M for zcu102 by default. There are two ways to modify CMA if needed
- Modify CONFIG_SUBSYSTEM_USER_CMDLINE option in `./project-spec/configs/config` file

- Modify by saving uboot environment on uboot command line. For example:
    ```
    ZynqMP> setenv bootargs "earlycon console=ttyPS0,115200 clk_ignore_unused root=/dev/mmcblk0p2 rw rootwait cma=512M"
    ZynqMP> saveenv
    Saving Environment to FAT... OK
    ZynqMP> reset
    resetting ..
    ```
    Check CMA when kernel starts up
    ```
    [    0.000000] cma: Reserved 512 MiB at 0x000000005fc00000
    ```

## Third-Party Content
All Xilinx and third-party licenses and sources associated with this reference design can be downloaded [here](https://www.xilinx.com/member/forms/download/xef.html?filename=xilinx-zynqmp-common-target-2022.1.tar.gz).


## License
Licensed under the Apache License, version 2.0 (the "License"); you may not use this file except in compliance with the License.

You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.


<hr/>
<p align="center"><sup>Copyright&copy; 2022 Xilinx</sup></p>
