import os

# Store all ".py" file names in the current directory and its subdirectories in a list
file_list = []
for root, dirs, files in os.walk("."):
    for file in files:
        if file.endswith(".py") or file.endswith(".ipynb"):
            file_list.append(os.path.join(root, file))

# Remove pathes containing "ipynb_checkpoints"
file_list = [x for x in file_list if "ipynb_checkpoints" not in x]
# Remove this file "Reconstruct.py" from the list
file_list.remove("./Reconstruct.py")

# Display files
print("Files to be reconstructed:")
for file in file_list:
    print(" - " + file)

# Deal with escape characters
safe_file_list = []
for file_name in file_list:
    safe_file_list.append(file_name.replace("(", '\(').replace(")", '\)'))

# Make a backup of all files
for file in safe_file_list:
    os.system("cp " + file + " " + file + ".bak")

# Reconstruction Table
recon_table = {

    # class ParamBasic:  # Fundamental parameters
    "config.basic['rootDir']": "config.basic.rootDir",
    "config.basic['train_DS_name']": "config.basic.train_DS_name",
    "config.basic['test_DS_name']": "config.basic.test_DS_name",
    "config.basic['normalization']": "config.basic.normalization",
    "config.basic['checkpointDir']": "config.basic.checkpointDir",
    "config.basic['tryID']": "config.basic.tryID",

    # class ParamModel:  # Parameters that controls model
    "config.model['checkpoint']": "config.model.checkpoint",
    "config.model['emb_dim']": "config.model.emb_dim",
    "config.model['gaze_backend_path']": "config.model.gaze_backend_path",
    "config.model['seed']": "config.model.seed",
    "config.model['leaky']": "config.model.leaky",
    "config.model['head_num']": "config.model.head_num",
    "config.model['dim_per_head']": "config.model.dim_per_head",
    "config.model['comp_dim']": "config.model.comp_dim",
    "config.model['mid_sizes']": "config.model.mid_sizes",
    "config.model['freeze_backend']": "config.model.freeze_backend",

    # class ParamOpti:  # Parameters that controls optimizing, loss, scheduling.
    "config.opti['lr']": "config.opti.lr",
    "config.opti['weight_decay']": "config.opti.weight_decay",
    "config.opti['backend_pct_start']": "config.opti.backend_pct_start",
    "config.opti['backend_div_factor']": "config.opti.backend_div_factor",
    "config.opti['classifier_pct_start']": "config.opti.classifier_pct_start",
    "config.opti['classifier_div_factor']": "config.opti.classifier_div_factor",
    "config.opti['enable_grad_clip']": "config.opti.enable_grad_clip",
    "config.opti['grad_clip_ref_range']": "config.opti.grad_clip_ref_range",
    "config.opti['grad_clip_mul']": "config.opti.grad_clip_mul",

    # class ParamLoss:  # Parameters that controls loss
    "config.loss['loss_func']": "config.loss.loss_func",
    "config.loss['gaze_weight']": "config.loss.gaze_weight",
    "config.loss['gaze_gamma']": "config.loss.gaze_gamma",

    # class ParamPrep:  # Parameters that controls preprocess
    "config.prep['thread_num']": "config.prep.thread_num",
    "config.prep['util_percent']": "config.prep.util_percent",
    "config.prep['tempDir']": "config.prep.tempDir",
    "config.prep['prep_every_epoch']": "config.prep.prep_every_epoch",
    "config.prep['save_dataset']": "config.prep.save_dataset",
    "config.prep['transform']": "config.prep.transform",
    "config.prep['argument']": "config.prep.argument",
    "config.prep['rand_horizontal_flip']": "config.prep.rand_horizontal_flip",
    "config.prep['trainCateDictDir']": "config.prep.trainCateDictDir",

    # class ParamTest:  # Parameters that controls test process
    "config.test['enable']": "config.test.enable",
    "config.test['batch_size']": "config.test.batch_size",
    "config.test['num_workers']": "config.test.num_workers",
    "config.test['util_percent']": "config.test.util_percent",
    "config.test['testClipsDir']": "config.test.testClipsDir",
    "config.test['resultDir']": "config.test.resultDir",
    "config.test['transform']": "config.test.transform",
    "config.test['verbose']": "config.test.verbose",
    "config.test['onlyEvalLastN']": "config.test.onlyEvalLastN",

    # class ParamTrain:  # Parameters that controls train process
    "config.train['enable']": "config.train.enable",
    "config.train['num_epochs']": "config.train.num_epochs",
    "config.train['batch_size']": "config.train.batch_size",
    "config.train['num_workers']": "config.train.num_workers",
    "config.train['seed']": "config.train.seed",
    "config.train['smooth_label_alpha']": "config.train.smooth_label_alpha",
    "config.train['smooth_label_beta']": "config.train.smooth_label_beta",
    "config.train['verbose']": "config.train.verbose",
}

# Replace old code with new code
for file in file_list:
    with open(file, "r") as f:
        code = f.read()
    for old_code, new_code in recon_table.items():
        code = code.replace(old_code, new_code)
    with open(file, "w") as f:
        f.write(code)

# print("Reconstruction finished.")

# Confirm the reconstruction, if the user confirms by inputting "y", the backup files will be deleted, and the reconstruction will be completed.
# Otherwise, the reconstruction will be reverted.
confirm = input("Confirm the reconstruction? (y/n): ")
if confirm == "y":
    for file in safe_file_list:
        os.system("rm " + file + ".bak")
    print("Backup files deleted.")
else:
    for file in safe_file_list:
        os.system("mv " + file + ".bak " + file)
    print("Reconstruction reverted.")
