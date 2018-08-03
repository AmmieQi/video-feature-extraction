import torch
from torch import nn
from model_3D import resnet, pre_act_resnet, wide_resnet, resnext, densenet

import pretrainedmodels
from pretrainedmodels import utils


def generate_3D_model(opt):
    assert opt.model in [
        'resnet', 'preresnet', 'wideresnet', 'resnext', 'densenet'
    ]
    use_fc = False
    if opt.model == 'resnet':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]

        from model_3D.resnet import get_fine_tuning_parameters

        if opt.model_depth == 10:
            model = resnet.resnet10(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                last_fc=use_fc)
        elif opt.model_depth == 18:
            model = resnet.resnet18(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                last_fc=use_fc)
        elif opt.model_depth == 34:
            model = resnet.resnet34(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                last_fc=use_fc)
        elif opt.model_depth == 50:
            model = resnet.resnet50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                last_fc=use_fc)
        elif opt.model_depth == 101:
            model = resnet.resnet101(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                last_fc=use_fc)
        elif opt.model_depth == 152:
            model = resnet.resnet152(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                last_fc=use_fc)
        elif opt.model_depth == 200:
            model = resnet.resnet200(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                last_fc=use_fc)
    elif opt.model == 'wideresnet':
        assert opt.model_depth in [50]

        from model_3D.wide_resnet import get_fine_tuning_parameters

        if opt.model_depth == 50:
            model = wide_resnet.resnet50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                k=opt.wide_resnet_k,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                last_fc=use_fc)
    elif opt.model == 'resnext':
        assert opt.model_depth in [50, 101, 152]

        from model_3D.resnext import get_fine_tuning_parameters

        if opt.model_depth == 50:
            model = resnext.resnet50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                last_fc=use_fc)
        elif opt.model_depth == 101:
            model = resnext.resnet101(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                last_fc=use_fc)
        elif opt.model_depth == 152:
            model = resnext.resnet152(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                last_fc=use_fc)
    elif opt.model == 'preresnet':
        assert opt.model_depth in [18, 34, 50, 101, 152, 200]

        from model_3D.pre_act_resnet import get_fine_tuning_parameters

        if opt.model_depth == 18:
            model = pre_act_resnet.resnet18(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                last_fc=use_fc)
        elif opt.model_depth == 34:
            model = pre_act_resnet.resnet34(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                last_fc=use_fc)
        elif opt.model_depth == 50:
            model = pre_act_resnet.resnet50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                last_fc=use_fc)
        elif opt.model_depth == 101:
            model = pre_act_resnet.resnet101(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                last_fc=use_fc)
        elif opt.model_depth == 152:
            model = pre_act_resnet.resnet152(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                last_fc=use_fc)
        elif opt.model_depth == 200:
            model = pre_act_resnet.resnet200(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                last_fc=use_fc)
    elif opt.model == 'densenet':
        assert opt.model_depth in [121, 169, 201, 264]

        from model_3D.densenet import get_fine_tuning_parameters

        if opt.model_depth == 121:
            model = densenet.densenet121(
                num_classes=opt.n_classes,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                last_fc=use_fc)
        elif opt.model_depth == 169:
            model = densenet.densenet169(
                num_classes=opt.n_classes,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                last_fc=use_fc)
        elif opt.model_depth == 201:
            model = densenet.densenet201(
                num_classes=opt.n_classes,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                last_fc=use_fc)
        elif opt.model_depth == 264:
            model = densenet.densenet264(
                num_classes=opt.n_classes,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                last_fc=use_fc)

    if not opt.no_cuda:
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=None)

        if opt.pretrain_path:
            print('loading pretrained model {}'.format(opt.pretrain_path))
            pretrain = torch.load(opt.pretrain_path)
            assert opt.arch == pretrain['arch']

            model.load_state_dict(pretrain['state_dict'])

            if opt.model == 'densenet':
                model.module.classifier = nn.Linear(
                    model.module.classifier.in_features, opt.n_finetune_classes)
                model.module.classifier = model.module.classifier.cuda()
            else:
                model.module.fc = nn.Linear(model.module.fc.in_features,
                                            opt.n_finetune_classes)
                model.module.fc = model.module.fc.cuda()

            parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
            return model, parameters
    else:
        if opt.pretrain_path:
            print('loading pretrained model {}'.format(opt.pretrain_path))
            pretrain = torch.load(opt.pretrain_path)
            assert opt.arch == pretrain['arch']

            model.load_state_dict(pretrain['state_dict'])

            if opt.model == 'densenet':
                model.classifier = nn.Linear(
                    model.classifier.in_features, opt.n_finetune_classes)
            else:
                model.fc = nn.Linear(model.fc.in_features,
                                            opt.n_finetune_classes)

            parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
            return model, parameters

    return model, model.parameters()


def generate_2D_model(opt):
    if opt['model'] == 'inception_v3':
        C, H, W = 3, 299, 299
        model = pretrainedmodels.inceptionv3(pretrained='imagenet')
        load_image_fn = utils.LoadTransformImage(model)
    elif opt['model'] == 'vgg16':
        C, H, W = 3, 224, 224
        model = pretrainedmodels.vgg16(pretrained='imagenet')
        load_image_fn = utils.LoadTransformImage(model)
    elif opt['model'] == 'vgg19':
        C, H, W = 3, 224, 224
        model = pretrainedmodels.vgg19(pretrained='imagenet')
        load_image_fn = utils.LoadTransformImage(model)
    elif opt['model'] == 'resnet50':
        C, H, W = 3, 224, 224
        model = pretrainedmodels.resnet50(pretrained='imagenet')
        load_image_fn = utils.LoadTransformImage(model)
    elif opt['model'] == 'resnet101':
        C, H, W = 3, 224, 224
        model = pretrainedmodels.resnet101(pretrained='imagenet')
        load_image_fn = utils.LoadTransformImage(model)
    elif opt['model'] == 'resnet152':
        C, H, W = 3, 224, 224
        model = pretrainedmodels.resnet152(pretrained='imagenet')
        load_image_fn = utils.LoadTransformImage(model)
    elif opt['model'] == 'inception_v4':
        C, H, W = 3, 299, 299
        model = pretrainedmodels.inceptionv4(
            num_classes=1000, pretrained='imagenet')
        load_image_fn = utils.LoadTransformImage(model)
    elif opt['model'] == 'nasnet':
        C, H, W = 3, 331, 331
        model = pretrainedmodels.nasnetalarge(num_classes=1001, pretrained='imagenet+background')
        load_image_fn = utils.LoadTransformImage(model)
    else:
        print("doesn't support %s" % (opt['model']))

    model.last_linear = utils.Identity()
    model = nn.DataParallel(model)
    # if opt['saved_model'] != '':
    #     model.load_state_dict(torch.load(opt['saved_model']), strict=False)
    model = model.cuda()
    return model