

from common_interface import model

from model_9.utils import SquadRunner


squad_runner = SquadRunner(dev_df, val_df, test_df_prod, train_batch_size=12, num_train_epochs=2, do_lower_case=True)