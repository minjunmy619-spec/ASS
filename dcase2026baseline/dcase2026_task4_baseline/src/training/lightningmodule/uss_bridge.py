from .base_lightningmodule import BaseLightningModule


class USSBridgeLightning(BaseLightningModule):
    """Opt-in USS lightning module for semantic-acoustic bridge training.

    The original USSLightning only forwards ``mixture`` to the model. Bridge USS
    models can optionally consume oracle ``spatial_vector`` during training for
    DoA/proposal supervision and scheduled spatial conditioning. This class keeps
    the old target contract and only forwards additional keys when they exist.
    """

    def _get_input_dict(self, batch_data_dict):
        input_dict = {"mixture": batch_data_dict["mixture"]}
        for key in ("spatial_vector", "spatial_clue", "doa_vector"):
            if key in batch_data_dict:
                input_dict[key] = batch_data_dict[key]
        if "spatial_vector" not in input_dict and "foreground_doa" in batch_data_dict:
            input_dict["spatial_vector"] = batch_data_dict["foreground_doa"]
        return input_dict

    def _get_target_dict(self, batch_data_dict):
        target_dict = {
            "mixture": batch_data_dict["mixture"],
            "foreground_waveform": batch_data_dict["foreground_waveform"],
            "interference_waveform": batch_data_dict["interference_waveform"],
            "noise_waveform": batch_data_dict["noise_waveform"],
            "class_index": batch_data_dict["class_index"],
            "is_silence": batch_data_dict["is_silence"],
        }
        for key in (
            "foreground_span_sec",
            "interference_span_sec",
            "noise_span_sec",
            "spatial_vector",
            "spatial_clue",
            "doa_vector",
            "foreground_doa",
            "foreground_doa_mask",
        ):
            if key in batch_data_dict:
                target_dict[key] = batch_data_dict[key]
        if "spatial_vector" not in target_dict and "foreground_doa" in batch_data_dict:
            target_dict["spatial_vector"] = batch_data_dict["foreground_doa"]
        return target_dict

    def training_step_processing(self, batch_data_dict, batch_idx):
        batchsize = batch_data_dict["mixture"].shape[0]
        output_dict = self.model(self._get_input_dict(batch_data_dict))
        target_dict = self._get_target_dict(batch_data_dict)
        loss_dict = self.loss_func(output_dict, target_dict)
        return batchsize, loss_dict

    def validation_step_processing(self, batch_data_dict, batch_idx):
        batchsize = batch_data_dict["mixture"].shape[0]
        output_dict = self.model(self._get_input_dict(batch_data_dict))
        target_dict = self._get_target_dict(batch_data_dict)
        loss_dict = self.loss_func(output_dict, target_dict)

        loss_dict = {k: v.item() for k, v in loss_dict.items()}
        if self.metric_func:
            metric = self.metric_func(output_dict, target_dict)
            for k, v in metric.items():
                loss_dict[k] = v.mean().item()

        return batchsize, loss_dict
