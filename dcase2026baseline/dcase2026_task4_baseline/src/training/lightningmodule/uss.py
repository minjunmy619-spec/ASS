from .base_lightningmodule import BaseLightningModule


class USSLightning(BaseLightningModule):
    def _get_target_dict(self, batch_data_dict):
        return {
            "foreground_waveform": batch_data_dict["foreground_waveform"],
            "interference_waveform": batch_data_dict["interference_waveform"],
            "noise_waveform": batch_data_dict["noise_waveform"],
            "class_index": batch_data_dict["class_index"],
            "is_silence": batch_data_dict["is_silence"],
        }

    def training_step_processing(self, batch_data_dict, batch_idx):
        batchsize = batch_data_dict["mixture"].shape[0]
        output_dict = self.model({"mixture": batch_data_dict["mixture"]})
        target_dict = self._get_target_dict(batch_data_dict)
        loss_dict = self.loss_func(output_dict, target_dict)
        return batchsize, loss_dict

    def validation_step_processing(self, batch_data_dict, batch_idx):
        batchsize = batch_data_dict["mixture"].shape[0]
        output_dict = self.model({"mixture": batch_data_dict["mixture"]})
        target_dict = self._get_target_dict(batch_data_dict)
        loss_dict = self.loss_func(output_dict, target_dict)

        loss_dict = {k: v.item() for k, v in loss_dict.items()}
        if self.metric_func:
            metric = self.metric_func(output_dict, target_dict)
            for k, v in metric.items():
                loss_dict[k] = v.mean().item()

        return batchsize, loss_dict
