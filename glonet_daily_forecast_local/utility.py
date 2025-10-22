from torchvision import transforms
import numpy as np


def get_normalizer1(model_dir: str):
    level = "L0"
    gmean = np.concatenate(
        [
            [np.load(model_dir + "/" + level + "/zos_mean.npy")],
            np.load(model_dir + "/" + level + "/thetao_mean.npy"),
            np.load(model_dir + "/" + level + "/so_mean.npy"),
            np.load(model_dir + "/" + level + "/uo_mean.npy"),
            np.load(model_dir + "/" + level + "/vo_mean.npy"),
        ]
    )

    gstd = np.concatenate(
        [
            [np.load(model_dir + "/" + level + "/zos_std.npy")],
            np.load(model_dir + "/" + level + "/thetao_std.npy"),
            np.load(model_dir + "/" + level + "/so_std.npy"),
            np.load(model_dir + "/" + level + "/uo_std.npy"),
            np.load(model_dir + "/" + level + "/vo_std.npy"),
        ]
    )

    transform = transforms.Normalize(mean=gmean, std=gstd)

    return transform


def get_denormalizer1(model_dir: str):
    level = "L0"
    gmean = np.concatenate(
        [
            [np.load(model_dir + "/" + level + "/zos_mean.npy")],
            np.load(model_dir + "/" + level + "/thetao_mean.npy"),
            np.load(model_dir + "/" + level + "/so_mean.npy"),
            np.load(model_dir + "/" + level + "/uo_mean.npy"),
            np.load(model_dir + "/" + level + "/vo_mean.npy"),
        ]
    )

    gstd = np.concatenate(
        [
            [np.load(model_dir + "/" + level + "/zos_std.npy")],
            np.load(model_dir + "/" + level + "/thetao_std.npy"),
            np.load(model_dir + "/" + level + "/so_std.npy"),
            np.load(model_dir + "/" + level + "/uo_std.npy"),
            np.load(model_dir + "/" + level + "/vo_std.npy"),
        ]
    )

    denormalizer = transforms.Normalize(
        mean=[-m / s for m, s in zip(gmean, gstd)], std=[1 / s for s in gstd]
    )
    return denormalizer


####################################################################3
################################################################################


def get_normalizer2(model_dir: str):
    levels = {
        "L50",
        "L100",
        "L150",
        "L222",
        "L318",
        "L380",
        "L450",
        "L540",
        "L640",
        "L763",
    }
    mean_thetao = []
    mean_so = []
    mean_uo = []
    mean_vo = []
    std_thetao = []
    std_so = []
    std_uo = []
    std_vo = []
    for level in sorted(levels):
        mean_thetao.append(
            np.load(model_dir + "/" + level + "/thetao_mean.npy")[0]
        )
        mean_so.append(np.load(model_dir + "/" + level + "/so_mean.npy")[0])
        mean_uo.append(np.load(model_dir + "/" + level + "/uo_mean.npy")[0])
        mean_vo.append(np.load(model_dir + "/" + level + "/vo_mean.npy")[0])

        std_thetao.append(
            np.load(model_dir + "/" + level + "/thetao_std.npy")[0]
        )
        std_so.append(np.load(model_dir + "/" + level + "/so_std.npy")[0])
        std_uo.append(np.load(model_dir + "/" + level + "/uo_std.npy")[0])
        std_vo.append(np.load(model_dir + "/" + level + "/vo_std.npy")[0])

    gmean = np.concatenate([mean_thetao, mean_so, mean_uo, mean_vo])
    gstd = np.concatenate([std_thetao, std_so, std_uo, std_vo])

    transform = transforms.Normalize(mean=gmean, std=gstd)

    return transform


def get_denormalizer2(model_dir: str):
    levels = {
        "L50",
        "L100",
        "L150",
        "L222",
        "L318",
        "L380",
        "L450",
        "L540",
        "L640",
        "L763",
    }
    mean_thetao = []
    mean_so = []
    mean_uo = []
    mean_vo = []
    std_thetao = []
    std_so = []
    std_uo = []
    std_vo = []
    for level in sorted(levels):
        mean_thetao.append(
            np.load(model_dir + "/" + level + "/thetao_mean.npy")[0]
        )
        mean_so.append(np.load(model_dir + "/" + level + "/so_mean.npy")[0])
        mean_uo.append(np.load(model_dir + "/" + level + "/uo_mean.npy")[0])
        mean_vo.append(np.load(model_dir + "/" + level + "/vo_mean.npy")[0])

        std_thetao.append(
            np.load(model_dir + "/" + level + "/thetao_std.npy")[0]
        )
        std_so.append(np.load(model_dir + "/" + level + "/so_std.npy")[0])
        std_uo.append(np.load(model_dir + "/" + level + "/uo_std.npy")[0])
        std_vo.append(np.load(model_dir + "/" + level + "/vo_std.npy")[0])

    gmean = np.concatenate([mean_thetao, mean_so, mean_uo, mean_vo])
    gstd = np.concatenate([std_thetao, std_so, std_uo, std_vo])

    denormalizer = transforms.Normalize(
        mean=[-m / s for m, s in zip(gmean, gstd)], std=[1 / s for s in gstd]
    )
    return denormalizer


####################################################################3
def get_normalizer3(model_dir: str):
    levels = {
        "L902",
        "L1245",
        "L1684",
        "L2225",
        "L3220",
        "L3597",
        "L3992",
        "L4405",
        "L4833",
        "L5274",
    }
    mean_thetao = []
    mean_so = []
    mean_uo = []
    mean_vo = []
    std_thetao = []
    std_so = []
    std_uo = []
    std_vo = []
    for level in sorted(levels):
        mean_thetao.append(
            np.load(model_dir + "/" + level + "/thetao_mean.npy")[0]
        )
        mean_so.append(np.load(model_dir + "/" + level + "/so_mean.npy")[0])
        mean_uo.append(np.load(model_dir + "/" + level + "/uo_mean.npy")[0])
        mean_vo.append(np.load(model_dir + "/" + level + "/vo_mean.npy")[0])

        std_thetao.append(
            np.load(model_dir + "/" + level + "/thetao_std.npy")[0]
        )
        std_so.append(np.load(model_dir + "/" + level + "/so_std.npy")[0])
        std_uo.append(np.load(model_dir + "/" + level + "/uo_std.npy")[0])
        std_vo.append(np.load(model_dir + "/" + level + "/vo_std.npy")[0])

    gmean = np.concatenate([mean_thetao, mean_so, mean_uo, mean_vo])
    gstd = np.concatenate([std_thetao, std_so, std_uo, std_vo])

    transform = transforms.Normalize(mean=gmean, std=gstd)

    return transform


def get_denormalizer3(model_dir: str):
    levels = {
        "L902",
        "L1245",
        "L1684",
        "L2225",
        "L3220",
        "L3597",
        "L3992",
        "L4405",
        "L4833",
        "L5274",
    }
    mean_thetao = []
    mean_so = []
    mean_uo = []
    mean_vo = []
    std_thetao = []
    std_so = []
    std_uo = []
    std_vo = []
    for level in sorted(levels):
        mean_thetao.append(
            np.load(model_dir + "/" + level + "/thetao_mean.npy")[0]
        )
        mean_so.append(np.load(model_dir + "/" + level + "/so_mean.npy")[0])
        mean_uo.append(np.load(model_dir + "/" + level + "/uo_mean.npy")[0])
        mean_vo.append(np.load(model_dir + "/" + level + "/vo_mean.npy")[0])

        std_thetao.append(
            np.load(model_dir + "/" + level + "/thetao_std.npy")[0]
        )
        std_so.append(np.load(model_dir + "/" + level + "/so_std.npy")[0])
        std_uo.append(np.load(model_dir + "/" + level + "/uo_std.npy")[0])
        std_vo.append(np.load(model_dir + "/" + level + "/vo_std.npy")[0])

    gmean = np.concatenate([mean_thetao, mean_so, mean_uo, mean_vo])
    gstd = np.concatenate([std_thetao, std_so, std_uo, std_vo])

    denormalizer = transforms.Normalize(
        mean=[-m / s for m, s in zip(gmean, gstd)], std=[1 / s for s in gstd]
    )
    return denormalizer
