import argparse
import config
from utils import prepare_dataset
from train_model import train_Omni
from model import Omni, OmniSmall
import wandb
from geo_resnet import ResNet1D, BasicBlock, BottleneckBlock
from datetime import datetime


parser = argparse.ArgumentParser(description='Omni')

parser.add_argument('-d', '--dataset',
                    type=str, default='nzer',
                    help='Dataset (NZER, GTMD, SGN, GEOD_OMS_YELP, GEOD_OSM_FSQ')
parser.add_argument('-r', '--region',
                    type=str, default='auck',
                    help='City or Region (NZER: auck, hope, norse, north, palm, GTMD: mel, sea, sin, tor, SGN: swiss,  GEOD_OSM_FSQ/GEOD_OSM_YELP: sin, edi, tor, pit)')
parser.add_argument('-m', '--model',
                    type=str, default='omni',
                    help='Omni or omnismall')
parser.add_argument('--run_att_aff',
                    action='store_true', help='Set flag to run attribute affinity')
parser.add_argument(
                    '--attributes',
                    nargs='+',  type=str, default=['name', 'type'],
                    help='List of attributes to compare with attribute affinity. For example, \'name type\' for nzer.')
parser.add_argument('--save_best_model',
                    action='store_true', help='Set flag to save best model')
parser.add_argument( '--save_path',
                    type=str, default='saved_models',
                    help='directory to save best model')
parser.add_argument('--log_wandb',
                    action='store_true', help='Set flag to enable wandb logging')

args = parser.parse_args()

device = config.device

multi_class= True if args.dataset=='GTMD' else False


# with open('./saved_models/best_model_my_datacombined-FINAL---2024-04-17-21-26-33_geo_embed_model', 'rb') as file:
#     geo_model = pickle.load(file)
geo_model = ResNet1D(BasicBlock, [2,2,2], 26, 512, dropout_rate=0.5)
geo_model = geo_model.to(device)

if args.model=="omnismall":
    model = OmniSmall(geo_model, multi_class, device=device, att_aff= True, dropout=config.dropout)
elif args.model=="omni":
    model = Omni(geo_model, multi_class, device=device, att_aff=args.run_att_aff, dropout=config.dropout)
else:
    print("Please choose a valid model between omni and omnismall")
    quit()



print("*****  Processing Train data  *****")
train_path = config.path + args.dataset + '/' + args.region + '/train.txt'
train_geom_path = config.path + args.dataset + '/' + args.region + '/train_geo.pkl'
train_x, train_coord, train_y, train_geo, train_min_dists, train_val_positions, train_cos_positions = prepare_dataset(train_path,
                                                                                                                      train_geom_path,
                                                                                                                      args.attributes,
                                                                                                                      args.run_att_aff,
                                                                                                                      max_seq_len=config.max_seq_len)

print("*****  Train data loaded  *****")
print("*****  Processing Validation data  *****")

valid_path = config.path + args.dataset + '/' + args.region + '/valid.txt'
valid_geom_path = config.path + args.dataset + '/' + args.region + '/valid_geo.pkl'
valid_x, valid_coord, valid_y, valid_geo, valid_min_dists, valid_val_positions, valid_cos_positions = prepare_dataset(valid_path,
                                                                                                                      valid_geom_path,
                                                                                                                      args.attributes,
                                                                                                                      args.run_att_aff,
                                                                                                                      max_seq_len=config.max_seq_len)

print("*****  Validation data loaded  *****")
print("*****  Processing Test data  *****")

test_path = config.path + args.dataset + '/' + args.region + '/test.txt'
test_geom_path = config.path + args.dataset + '/' + args.region + '/test_geo.pkl'
test_x, test_coord, test_y, test_geo, test_min_dists, test_val_positions, test_cos_positions = prepare_dataset(test_path,
                                                                                                               test_geom_path,
                                                                                                               args.attributes,
                                                                                                               args.run_att_aff,
                                                                                                               max_seq_len=config.max_seq_len)

print("*****  Test data loaded  *****")


print('Loaded', args.region, ' from ', args.dataset, 'dataset')
print('Train size:',len(train_x))
print('Valid size:',len(valid_x))
print('Test size:',len(test_x))

assert len(train_x) == train_geo['geoms_left'].shape[0] == train_geo['geoms_right'].shape[0]
assert len(valid_x) == valid_geo['geoms_left'].shape[0] == valid_geo['geoms_right'].shape[0]
assert len(test_x) == test_geo['geoms_left'].shape[0] == test_geo['geoms_right'].shape[0]


model = model.to(device)



current_timestamp = datetime.now()
timestamp_string = current_timestamp.strftime("%Y-%m-%d-%H-%M-%S")
omni_added = "-attribute-affinity-"
if args.log_wandb:
    wandb.init(
          project = "Omni-Geometry-Entity-Resolution",
          name=f"name-type-"+omni_added + timestamp_string,
          config={
          "learning_rate": 3e-5,
          "architecture": "Added omni geometry encoder",
          "dataset": args.dataset+'-'+args.region,
          "epochs": config.epochs,
          })

# save_path=config.save_path+args.dataset.lower()+args.region.lower()+omni_added+'--'+timestamp_string
train_Omni(model,
           train_x, train_coord, train_geo, train_min_dists, train_val_positions, train_cos_positions,
           train_y, valid_x, valid_coord, valid_geo, valid_min_dists, valid_val_positions, valid_cos_positions, valid_y,
           test_x, test_coord, test_geo, test_min_dists, test_val_positions, test_cos_positions, test_y,
           args.log_wandb, device, args.save_best_model, save_path=args.save_path,
           epochs=config.epochs, batch_size=config.batch_size, lr=config.lr)

if args.log_wandb:
    wandb.finish()