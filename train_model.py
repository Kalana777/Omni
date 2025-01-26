import torch
from torch import nn, optim
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np
import pickle
import wandb
from metrics import calculate_metrics, calculate_metrics2
import time


def train_Omni(model, train_x, train_coord, train_geo, train_min_dists, train_val_positions, train_cos_positions,
               train_y, valid_x, valid_coord, valid_geo, valid_min_dists, valid_val_positions, valid_cos_positions,
               valid_y, test_x, test_coord, test_geo, test_min_dists, test_val_positions, test_cos_positions, test_y,
               wandb_logging, device, save_best_model, save_path,  epochs=10, batch_size=32, lr=3e-5):

    opt = AdamW(params=model.parameters(), lr=lr)
    loss_function = nn.NLLLoss()

    valid_x_tensor = torch.tensor(valid_x)
    valid_coord_tensor = torch.tensor(valid_coord)
    valid_min_dists = torch.tensor(valid_min_dists)
    valid_y_tensor = torch.tensor(valid_y)

    test_x_tensor = torch.tensor(test_x)
    test_coord_tensor = torch.tensor(test_coord)
    test_min_dists = torch.tensor(test_min_dists)
    test_y_tensor = torch.tensor(test_y)

    num_steps = (len(train_x) // batch_size) * epochs
    scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=0, num_training_steps=num_steps)

    best_f1 = 0.0

    model.train()
    model.geo_embed_model.train()

    for epoch in range(epochs):

        print('\n*** TRAINING EPOCH:', epoch + 1, '***\n')

        i = 0
        step = 1

        while i < len(train_x):

            opt.zero_grad()

            if i + batch_size > len(train_x):
                y = train_y[i:]
                x = train_x[i:]
                x_coord = train_coord[i:]
                x_min_dists = train_min_dists[i:]
                x_val_pos = train_val_positions[i:]
                # x_cos_pos = {"names":train_cos_positions["names"][i:], "types":train_cos_positions["types"][i:],"addresses":train_cos_positions["addresses"][i:]}
                # x_cos_pos = {"names":train_cos_positions["names"][i:],"addresses":train_cos_positions["addresses"][i:]}
                x_cos_pos = {'attribute1': train_cos_positions['attribute1'][i:],
                             'attribute2': train_cos_positions['attribute2'][i:]}
                x_geo_left = train_geo['geoms_left'][i:]
                x_geo_right = train_geo['geoms_right'][i:]
                x_geo_left_type = train_geo['type_left'][i:]
                x_geo_right_type = train_geo['type_right'][i:]


            else:
                y = train_y[i: i + batch_size]
                x = train_x[i: i + batch_size]
                x_coord = train_coord[i: i + batch_size]
                x_min_dists = train_min_dists[i: i + batch_size]
                x_val_pos = train_val_positions[i: i + batch_size]
                # x_cos_pos = train_cos_positions[i : i + batch_size]
                # x_cos_pos = {"names": train_cos_positions["names"][i : i + batch_size], "types": train_cos_positions["types"][i : i + batch_size],  "addresses": train_cos_positions["addresses"][i: i + batch_size]}
                x_cos_pos = {'attribute1': train_cos_positions['attribute1'][i: i + batch_size],
                             'attribute2': train_cos_positions['attribute2'][i: i + batch_size]}
                x_geo_left = train_geo['geoms_left'][i: i + batch_size]
                x_geo_right = train_geo['geoms_right'][i: i + batch_size]
                x_geo_left_type = train_geo['type_left'][i: i + batch_size]
                x_geo_right_type = train_geo['type_right'][i: i + batch_size]

            y = torch.tensor(y).view(-1).to(device)
            x = torch.tensor(x)
            x_coord = torch.tensor(x_coord).to(device)
            x_min_dists = torch.tensor(x_min_dists).to(device)
            att_mask = torch.tensor(np.where(x != 0, 1, 0)).to(device)
            x_geo_left = x_geo_left.to(device)
            x_geo_right = x_geo_right.to(device)

            # x_val_pos = torch.tensor(x_val_pos)

            pred = model(x, x_coord, x_min_dists, att_mask, x_val_pos, x_cos_pos, x_geo_left, x_geo_right)

            loss = loss_function(pred, y)
            loss.backward()
            opt.step()


            if step % 10 == 0:
                if device == 'cuda':
                    print('Step:', step, 'Loss:', loss.cpu().detach().numpy())
                else:
                    print('Step:', step, 'Loss:', loss)

            step += 1
            scheduler.step()
            i += batch_size

        print('\n*** Validation Epoch:', epoch + 1, '***\n')
        val_f1, val_acc, val_prec, val_recall, _ = validate_model(model, valid_x_tensor, valid_coord_tensor,
                                                                  valid_min_dists, valid_val_positions,
                                                                  valid_cos_positions, valid_y_tensor, valid_geo,
                                                                  device)

        print('\n*** Test Epoch:', epoch + 1, '***\n')
        test_f1, test_acc, test_prec, test_recall, _ = validate_model(model, test_x_tensor, test_coord_tensor,
                                                                      test_min_dists, test_val_positions,
                                                                      test_cos_positions, test_y_tensor, test_geo,
                                                                      device)

        if wandb_logging:
            wandb.log(
                {"val_f1": val_f1, "loss": loss, "val_acc": val_acc, "val_prec": val_prec, "val_recall": val_recall,
                 "test_f1": test_f1, "test_acc": test_acc, "test_prec": test_prec, "test_recall": test_recall})

        if save_best_model:
            if val_f1 > best_f1:
              best_f1 = val_f1

              torch.save(model.state_dict(), save_path+'\\omni.pth')
              # torch.save(model.geo_embed_model.state_dict(), 'D:\Omni_Geo_ER\\models_for_pred\\omni_no_aff_geo_embed.pth')
              # torch.save(model.language_model.state_dict(), 'D:\Omni_Geo_ER\\models_for_pred\\omni_no_aff_lm.pth')



def validate_model(model, valid_x_tensor, valid_coord_tensor, valid_min_dists, valid_val_pos, valid_cos_pos,
                   valid_y_tensor, valid_geo, device, batch_size=512):

    model.eval()
    model.geo_embed_model.eval()
    start_time_test = time.time()
    with torch.no_grad():
        all_predictions = []
        i = 0
        length_valid_x = valid_x_tensor.shape[0]

        while i < length_valid_x:

            if i + batch_size > length_valid_x:
                x = valid_x_tensor[i:]
                x_coord = valid_coord_tensor[i:]
                x_min_dists = valid_min_dists[i:]
                x_val_pos = valid_val_pos[i:]
                # x_cos_pos = {"names":valid_cos_pos["names"][i:], "types":valid_cos_pos["types"][i:],"addresses":valid_cos_pos["addresses"][i:]}
                # x_cos_pos = {"names":valid_cos_pos["names"][i:], "addresses":valid_cos_pos["addresses"][i:]}
                x_cos_pos = {'attribute1': valid_cos_pos['attribute1'][i:],
                             'attribute2': valid_cos_pos['attribute2'][i:]}
                # x_n = valid_n[i:]
                x_geo_left = valid_geo['geoms_left'][i:]
                x_geo_right = valid_geo['geoms_right'][i:]
                x_geo_left_type = valid_geo['type_left'][i:]
                x_geo_right_type = valid_geo['type_right'][i:]


            else:
                x = valid_x_tensor[i: i + batch_size]
                x_coord = valid_coord_tensor[i: i + batch_size]
                x_min_dists = valid_min_dists[i: i + batch_size]
                x_val_pos = valid_val_pos[i: i + batch_size]
                # x_cos_pos = valid_cos_pos[i: i + batch_size]
                # x_cos_pos = {"names": valid_cos_pos["names"][i: i + batch_size], "types": valid_cos_pos["types"][i: i + batch_size], "addresses": valid_cos_pos["addresses"][i: i + batch_size]}
                # x_cos_pos = {"names": valid_cos_pos["names"][i: i + batch_size],  "addresses": valid_cos_pos["addresses"][i: i + batch_size]}
                x_cos_pos = {'attribute1': valid_cos_pos['attribute1'][i: i + batch_size],
                             'attribute2': valid_cos_pos['attribute2'][i: i + batch_size]}
                x_geo_left = valid_geo['geoms_left'][i: i + batch_size]
                x_geo_right = valid_geo['geoms_right'][i: i + batch_size]
                x_geo_left_type = valid_geo['type_left'][i: i + batch_size]
                x_geo_right_type = valid_geo['type_right'][i: i + batch_size]

            x = x.to(device)
            x_coord = x_coord.to(device)
            x_min_dists = x_min_dists.to(device)
            att_mask = torch.tensor(np.where(x.cpu() != 0, 1, 0)).to(device)
            x_geo_left = x_geo_left.to(device)
            x_geo_right = x_geo_right.to(device)
            x_geo_left_type = x_geo_left_type.to(device)
            x_geo_right_type = x_geo_right_type.to(device)

            # x_val_pos = torch.tensor(x_val_pos)

            predictions = model(x, x_coord, x_min_dists, att_mask, x_val_pos, x_cos_pos, x_geo_left, x_geo_right,
                                training=False)
            all_predictions.append(predictions)

            i += batch_size

    end_time_test = time.time()
    print(f"Prediction time: {end_time_test - start_time_test}s for {length_valid_x} examples")
    all_predictions = torch.cat(all_predictions, dim=0)

    if model.multi_class:
        metrics = calculate_metrics2(all_predictions, valid_y_tensor)
    else:
        metrics = calculate_metrics(all_predictions, valid_y_tensor)

    print("Accuracy:", metrics['accuracy'])
    print("Precision:", metrics['precision'])
    print("Recall:", metrics['recall'])
    print("f1-Score:", metrics['f1'])

    return metrics['f1'], metrics['accuracy'], metrics['precision'], metrics['recall'], torch.argmax(all_predictions,
                                                                                                     dim=1)
