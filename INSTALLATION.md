# Some Notes on Installation

Notice that all the operations are taken under the project's conda environment. Be sure to activate it in case omitted.

## cmake
```bash
pip install cmake==3.29.0.1
```

## CUDNN
```bash
conda install -n <YOUR_CONDA_ENV_NAME> cudnn
```
You may need to specify the `LD_LIBRARY_PATH` for `lake exe LeanCopilot/download`. An example command is:
```bash
export LD_LIBRARY_PATH="~/anaconda3/envs/<YOUR_CONDA_ENV_NAME>/lib/python3.10/site-packages/nvidia/cudnn/lib/:$LD_LIBRARY_PATH"
```
<!-- export LD_LIBRARY_PATH="/localscratch/cat/anaconda3/envs/dojo/lib/python3.10/site-packages/nvidia/cudnn/lib/:$LD_LIBRARY_PATH" -->

### elan
```bash
# Get the bash script
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh

# Add the path to the bash script
echo 'export PATH="$HOME/.elan/bin:$PATH"' >> ~/.bashrc
source $HOME/.elan/env
source ~/.bashrc
```

### Git-LFS
To install Git LFS on a remote server without `sudo` permissions:

```bash
# 1. Define the working directory where Git LFS will be installed.
WORKING_DIR=${WORKING_DIR}  # Replace ${WORKING_DIR} with the desired directory
mkdir -p $WORKING_DIR/.local/bin
cd $WORKING_DIR

# 2. Download the Git LFS binary for Linux and extract the files
wget https://github.com/git-lfs/git-lfs/releases/download/v3.4.1/git-lfs-linux-amd64-v3.4.1.tar.gz
tar xvf git-lfs-linux-amd64-v3.4.1.tar.gz

# 3. Navigate to the extracted directory and modify the prefix for installation
cd git-lfs-3.4.1/
chmod +x install.sh
vi install.sh
# Modify the prefix from /usr/local/ to $WORKING_DIR/.local/bin
# This step is manual

# 4. Add the installation directory to PATH
echo 'export PATH="'$WORKING_DIR'/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# 5. Run the installation script.
./install.sh

# 6. Verify the Git LFS installation.
git-lfs version
```

### SMT-Portfolio
```bash
pip install smt-portfolio==0.3.3
```

### Z3-Solver
```bash
pip install z3-solver==4.13.0.0
```
```bash
# Get the source repository
cd $WORKING_DIR
git clone https://github.com/Z3Prover/z3.git
cd z3

# Build z3
python scripts/mk_make.py
cd build; make  # This can take half an hour
make install

# Update the path; be sure to modify it to your local path
echo 'export PATH="'$your_conda_env_path'/bin:$PATH"' >> ~/.bashrc
echo 'export PYTHONPATH="'$your_conda_env_path'/lib/python3.x/site-packages:$PYTHONPATH"' >> ~/.bashrc
source ~/.bashrc
```
We may also use `cmake` to build:
```bash
# Clean the source
cd ../../z3  # i.e., the root folder for z3
git clean -nx src
git clean -fx src

# Build with cmake
mkdir build  # You may need to rm -rf build first
cd build
cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX=$your_conda_env_path ../
make -j$(nproc)
make install  # This will install z3 to the conda environment
```

### CVC5
### (prerequisite) m4
```bash
# Get the binary
cd $WORKING_DIR
wget https://ftp.gnu.org/gnu/m4/m4-latest.tar.gz
tar -xvf m4-latest.tar.gz
cd m4-1.4.19  # adjust version number as needed.

# Build the binary
./configure --prefix=$WORKING_DIR/m4
make
make install

# Update the path; be sure to modify the working dir
echo 'export PATH="'$WORKING_DIR'/m4/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Check if it works
m4 --version
```

### CVC5
```bash
# Clone the repo
cd ~/github
git clone https://github.com/cvc5/cvc5.git
cd cvc5

# Configure to the build
./configure.sh --prefix=$WORKING_DIR --auto-download

#
cd build
make -j$(nproc)

# Check
make check

# Install
make install

# Update the path; be sure to modify the working dir
echo 'export PATH="'$WORKING_DIR'/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### Vampire
```bash
# Get the repo
cd ~/github
git clone git@github.com:vprover/vampire.git
cd vampire

# Init submodule for z3
git submodule update --init

# make a clean directory to build Vampire into
mkdir build && cd build

# Configure the build and generate files with CMake
cmake .. -DCMAKE_INSTALL_PREFIX=$WORKING_DIR

# Build Vampire, in this case with make(1)
make -j$(nproc)

# Build Z3 into vampire/z3/build
mkdir -p ../z3/build && cd ../z3/build

# Configure single-threaded build, no debug symbols
cmake .. -DZ3_SINGLE_THREADED=1 -DCMAKE_BUILD_TYPE=Release

# Build Z3, in this case with make(1)
make -j$(nproc)

# Since it does not have an install target, we can give an alias to it
# The exact path may be modified
echo "alias vampire='~/github/vampire/build/bin/vampire_rel_master_7397'" >> ~/.bashrc
```

