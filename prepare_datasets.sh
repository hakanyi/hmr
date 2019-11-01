# ---------------------------
# ----- SET YOUR PATH!! -----
# ---------------------------

DATA_DIR=/gpfs/milgram/project/yildirim/hakan/datasets

# This is where you want all of your tf_records to be saved:
TF_DIR="$DATA_DIR/hmr_release_files/test_tf_datasets"

LSP_DIR="$DATA_DIR/lsp"

LSP_EXT_DIR="$DATA_DIR/lsp_ext"

MPII_DIR="$DATA_DIR/mpii"

COCO_DIR="$DATA_DIR/coco"

MPI_INF_3DHP_DIR="$DATA_DIR/mpi_inf_3dhp"

## Mosh
# This is the path to the directory that contains neutrSMPL_* directories
MOSH_DIR="$DATA_DIR/neutrMosh"
# ---------------------------


# ---------------------------
# Run each command below from this directory. I advice to run each one independently.
# ---------------------------

# ----- LSP -----
python -m src.datasets.lsp_to_tfrecords --img_directory $LSP_DIR --output_directory $TF_DIR/lsp

# ----- LSP-extended -----
python -m src.datasets.lsp_to_tfrecords --img_directory $LSP_EXT_DIR --output_directory $TF_DIR/lsp_ext

# ----- MPII -----
python -m src.datasets.mpii_to_tfrecords --img_directory $MPII_DIR --output_directory $TF_DIR/mpii

# ----- COCO -----
python -m src.datasets.coco_to_tfrecords --data_directory $COCO_DIR --output_directory $TF_DIR/coco

# ----- MPI-INF-3DHP -----
python -m src.datasets.mpi_inf_3dhp_to_tfrecords --data_directory $MPI_INF_3DHP_DIR --output_directory $TF_DIR/mpi_inf_3dhp

# ----- Mosh data, for each dataset -----
# CMU:
python -m src.datasets.smpl_to_tfrecords --data_directory $MOSH_DIR --output_directory $TF_DIR/mocap_neutrMosh --dataset_name 'neutrSMPL_CMU'

# H3.6M:
python -m src.datasets.smpl_to_tfrecords --data_directory $MOSH_DIR --output_directory $TF_DIR/mocap_neutrMosh --dataset_name 'neutrSMPL_H3.6'

# jointLim:
python -m src.datasets.smpl_to_tfrecords --data_directory $MOSH_DIR --output_directory $TF_DIR/mocap_neutrMosh --dataset_name 'neutrSMPL_jointLim'
