# Some Notes on Installation

## cmake
```bash
pip install cmake==3.29.0.1
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
# conda activate rir
pip install smt-portfolio==0.3.3
```

### Z3-Solver
```bash
# conda activate rir
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

# Update the path
echo 'export PATH="/nethome/hsun409/anaconda3/envs/dojo/bin:$PATH"' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH="/nethome/hsun409/anaconda3/envs/dojo/lib:$LD_LIBRARY_PATH"' >> ~/.bashrc
echo 'export PYTHONPATH="/nethome/hsun409/anaconda3/envs/dojo/lib/python3.x/site-packages:$PYTHONPATH"' >> ~/.bashrc
source ~/.bashrc
# conda activate rir
```

### CVC5
### (prerequisite) m4
```bash
# Get the binary
wget https://ftp.gnu.org/gnu/m4/m4-latest.tar.gz
tar -xvf m4-latest.tar.gz
cd m4-1.4.19  # adjust version number as needed.

# Build the binary
./configure --prefix=/localscratch/hsun409/m4
make
make install

# Update the path
echo 'export PATH="/localscratch/hsun409/m4/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
# conda activate rir

# Check if it works
m4 --version
```

### CVC5
```bash
# Clone the repo
git clone https://github.com/cvc5/cvc5.git
cd cvc5

# Configure to the build
./configure.sh --prefix=/localscratch/hsun409 --auto-download

#
cd build
make -j$(nproc)

# Check
make check

# Install
make install

# Update the path
echo 'export PATH="/localscratch/hsun409/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

