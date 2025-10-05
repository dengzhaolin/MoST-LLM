import torch
import numpy as np
def apply_rand_mask(data, mask_type="temporal", spatial_points=50, temporal_hours=2, return_mask=False):

    B,L,N,M,C = data.shape
    masked_data = data.clone()
    mask = torch.ones_like(data)

    if mask_type == "temporal":

        masked_channel = np.random.randint(0, M)

        mask_time_steps = np.random.choice(L, temporal_hours, replace=False)

        masked_data[:, mask_time_steps, :, masked_channel, :] = 0
        mask[:, mask_time_steps, :, masked_channel, :] = 0

    elif mask_type == "spatial":

        masked_channel = np.random.randint(0, M)


        mask_saptial= np.random.choice(N, spatial_points, replace=False)

        masked_data[:, :, mask_saptial, masked_channel,:] = 0
        mask[:, :, mask_saptial, masked_channel, :] = 0

    elif mask_type == "agnostic":

        if np.random.random() > 0.5:
            masked_data, mask = apply_mask(masked_data, mask_type="temporal",
                                           spatial_points=spatial_points, temporal_hours=temporal_hours, return_mask=True)
        else:
            masked_data, mask = apply_mask(masked_data, mask_type="spatial",
                                           spatial_points=spatial_points, temporal_hours=temporal_hours, return_mask=True)

    if return_mask:
        return masked_data, mask
    return masked_data


def apply_mask_single(data, mask_type="temporal",  spatial_points=20, temporal_hours=2, return_mask=False):
    B, L, N, M, C = data.shape
    masked_data = data.clone()
    mask = torch.ones_like(data)

    for channel in range(M):
        if mask_type == "temporal":

            mask_time_steps = np.random.choice(L, temporal_hours, replace=False)
            masked_data[:, mask_time_steps, :, channel,  :] = 0
            mask[:, mask_time_steps, :, channel,  :] = 0


        elif mask_type == "spatial":

            mask_saptial = np.random.choice(N, spatial_points, replace=False)

            masked_data[:, :, mask_saptial, channel, :] = 0

            mask[:, :, mask_saptial, channel, :] = 0

        elif mask_type == "agnostic":

            if np.random.random() > 0.5:
                temp_masked, temp_mask = apply_mask_single(
                    data[:, :, :, channel, :].unsqueeze(3),
                    mask_type="temporal",
                    temporal_hours=temporal_hours,
                    return_mask=True
                )
                masked_data[:, :, :, channel,  :] = temp_masked.squeeze(3)
                mask[:, :, :, channel, :] = temp_mask.squeeze(3)
            else:
                temp_masked, temp_mask = apply_mask_single(
                    data[:, :, :, channel, :].unsqueeze(3),
                    mask_type="spatial",
                    spatial_points=spatial_points,
                    return_mask=True
                )
                masked_data[:, :, :, channel,  :] = temp_masked.squeeze(3)
                mask[:, :, :, channel, :] = temp_mask.squeeze(3)

    if return_mask:
        return masked_data, mask
    return masked_data


def apply_mask(data, mask_type="temporal", spatial_points=50, temporal_hours=2, return_mask=False):

    B,L,N,M,C = data.shape
    masked_data = data.clone()
    mask = torch.ones_like(data)

    if mask_type == "temporal":

        masked_channel = np.random.randint(0, M)

        mask_time_steps = np.random.choice(L, temporal_hours, replace=False)

        masked_data[:, mask_time_steps, :, masked_channel, :] = 0
        mask[:, mask_time_steps, :, masked_channel, :] = 0

    elif mask_type == "spatial":

        masked_channel = np.random.randint(0, M)


        mask_saptial= np.random.choice(N, spatial_points, replace=False)

        masked_data[:, :, mask_saptial, masked_channel,:] = 0
        mask[:, :, mask_saptial, masked_channel, :] = 0

    elif mask_type == "agnostic":

        if np.random.random() > 0.5:
            masked_data, mask = apply_mask(masked_data, mask_type="temporal",
                                           spatial_points=spatial_points, temporal_hours=temporal_hours, return_mask=True)
        else:
            masked_data, mask = apply_mask(masked_data, mask_type="spatial",
                                           spatial_points=spatial_points, temporal_hours=temporal_hours, return_mask=True)

    if return_mask:
        return masked_data, mask
    return masked_data
def apply_mask_single_different(data, mask_type="temporal",  spatial_points=20, temporal_hours=2, return_mask=False):
    B, L, N, M, C = data.shape
    masked_data = data.clone()
    mask = torch.ones_like(data)

    for channel in range(M):

        if channel>=2:
            spatial_points=25
            temporal_hours=4

        if mask_type == "temporal":

            mask_time_steps = np.random.choice(L, temporal_hours, replace=False)
            masked_data[:, mask_time_steps, :, channel,  :] = 0
            mask[:, mask_time_steps, :, channel,  :] = 0


        elif mask_type == "spatial":


            mask_saptial = np.random.choice(N, spatial_points, replace=False)

            masked_data[:, :, mask_saptial, channel, :] = 0

            mask[:, :, mask_saptial, channel, :] = 0

        elif mask_type == "agnostic":

            if np.random.random() > 0.5:
                temp_masked, temp_mask = apply_mask_single(
                    data[:, :, :, channel, :].unsqueeze(3),
                    mask_type="temporal",
                    temporal_hours=temporal_hours,
                    return_mask=True
                )
                masked_data[:, :, :, channel,  :] = temp_masked.squeeze(3)
                mask[:, :, :, channel, :] = temp_mask.squeeze(3)
            else:
                temp_masked, temp_mask = apply_mask_single(
                    data[:, :, :, channel, :].unsqueeze(3),
                    mask_type="spatial",
                    spatial_points=spatial_points,
                    return_mask=True
                )
                masked_data[:, :, :, channel,  :] = temp_masked.squeeze(3)
                mask[:, :, :, channel, :] = temp_mask.squeeze(3)

    if return_mask:
        return masked_data, mask
    return masked_data