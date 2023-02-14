import numpy as np
import pandas as pd

NER_INFERENCE_DATA = [
    "what is SEMRUSH PRO? Can you run complex queries ? Can you identify "
    "active usage ?",
    "Thank you for your subscription renewal",
    "you can upgrade your account for an old price,while you can upgrade your "
    "account for $399.95/month",
    "I like EMSI ordered the pro package",
    "Here you go, your account is created",
]


NER_INFERENCE_PRED_TOKEN_SPANS = [
    [
        {
            "start": 4,
            "end": 7,
            "label": "Questions About the Product",
        },
        {
            "start": 12,
            "end": 14,
            "label": "Product Usage",
        },
    ],
    [
        {
            "start": 4,
            "end": 6,
            "label": "Renew",
        },
    ],
    [
        {
            "start": 0,
            "end": 3,
            "label": "Questions About the Product",
        },
        {
            "start": 15,
            "end": 19,
            "label": "Potential Upsell",
        },
    ],
    [],  # No spans
    [
        {
            "start": 0,
            "end": 2,
            "label": "Action item accomplished",
        },
        {
            "start": 7,
            "end": 8,
            "label": "Action item accomplished",
        },
    ],
]


class TestSpacyInfExpectedResults:
    _num_pred_spans = 7

    gt_data = pd.DataFrame(
        data={
            "id": [0, 1, 2, 3, 4],
            "split": ["inference"] * 5,
            "text": [text for text in NER_INFERENCE_DATA],
            "text_token_indices": [
                np.array(
                    [
                        0,
                        4,
                        5,
                        7,
                        8,
                        15,
                        16,
                        19,
                        19,
                        20,
                        21,
                        24,
                        25,
                        28,
                        29,
                        32,
                        33,
                        40,
                        41,
                        48,
                        49,
                        50,
                        51,
                        54,
                        55,
                        58,
                        59,
                        67,
                        68,
                        74,
                        75,
                        80,
                        81,
                        82,
                    ]
                ),
                np.array([0, 5, 6, 9, 10, 13, 14, 18, 19, 31, 32, 39]),
                np.array(
                    [
                        0,
                        3,
                        4,
                        7,
                        8,
                        15,
                        16,
                        20,
                        21,
                        28,
                        29,
                        32,
                        33,
                        35,
                        36,
                        39,
                        40,
                        45,
                        45,
                        46,
                        46,
                        51,
                        52,
                        55,
                        56,
                        59,
                        60,
                        67,
                        68,
                        72,
                        73,
                        80,
                        81,
                        84,
                        85,
                        86,
                        86,
                        92,
                        92,
                        93,
                        93,
                        98,
                    ]
                ),
                np.array([0, 1, 2, 6, 7, 11, 12, 19, 20, 23, 24, 27, 28, 35]),
                np.array([0, 4, 5, 8, 9, 11, 11, 12, 13, 17, 18, 25, 26, 28, 29, 36]),
            ],
            "data_schema_version": [1] * 5,
            "inference_name": ["inf-name"] * 5,
        }
    )

    gt_embs = np.array(
        [
            [
                2.10308933e00,
                -6.71794713e-01,
                7.24776328e-01,
                -1.99953222e00,
                -1.15847349e-01,
                1.05597043e00,
                2.25010252e00,
                8.02300572e-01,
                2.37572527e00,
                1.81910670e00,
                1.06117749e00,
                1.57220340e00,
                1.18388128e00,
                1.22505903e00,
                6.74207211e-01,
                -4.26268578e-01,
                3.02831173e00,
                2.25635791e00,
                1.60561061e00,
                3.15114617e-01,
                1.49914622e00,
                3.97937894e-02,
                1.40396690e00,
                -7.34017909e-01,
                1.55272579e00,
                1.28562951e00,
                2.82228446e00,
                8.62070858e-01,
                -1.12293959e00,
                7.56662548e-01,
                1.23021436e00,
                1.65270221e00,
                -1.39169097e00,
                1.83697081e00,
                -2.85280466e-01,
                1.77106237e00,
                -2.70398736e-01,
                2.36902952e00,
                1.34924555e00,
                1.38026118e00,
                1.99271739e00,
                1.40931678e00,
                1.79851055e00,
                2.60265493e00,
                -5.10122299e-01,
                3.65344644e-01,
                9.17914331e-01,
                2.72919893e00,
                2.83239031e00,
                5.66840172e-03,
                1.99800968e00,
                4.27378893e-01,
                -1.21140122e00,
                1.27158880e-01,
                2.53665268e-01,
                8.74667466e-01,
                -1.04060566e00,
                -4.45200384e-01,
                1.12860215e00,
                1.42238843e00,
                -1.09184372e00,
                3.09182477e00,
                4.09180880e-01,
                -1.08000445e00,
            ],
            [
                2.98322845e00,
                1.93407685e-01,
                3.89509416e00,
                -8.63219023e-01,
                2.51305252e-01,
                4.89649445e-01,
                1.26431441e00,
                -1.54169369e00,
                1.73743665e00,
                -2.26657534e00,
                3.14958763e00,
                2.15655833e-01,
                -1.12570834e00,
                1.51168478e00,
                -9.25579667e-01,
                1.51415968e00,
                3.27247000e00,
                3.39723635e00,
                5.53597975e00,
                2.96228623e00,
                -4.81992453e-01,
                -6.13452673e-01,
                2.49668384e00,
                1.38363063e-01,
                -2.62995839e00,
                -1.32838893e00,
                6.13844752e-01,
                1.61423194e00,
                -2.00744414e00,
                1.64176464e00,
                8.56476784e-01,
                1.20557219e-01,
                4.15441304e-01,
                2.01010734e-01,
                2.92484975e00,
                1.01773214e00,
                -7.35310912e-01,
                2.48402810e00,
                2.00613046e00,
                1.49489951e00,
                4.93443632e00,
                1.71129608e00,
                -9.90877032e-01,
                2.28111577e00,
                2.56828308e00,
                3.20763159e00,
                7.61784792e-01,
                7.03613460e-02,
                4.51654005e00,
                3.83612370e00,
                1.59185398e00,
                -6.69722259e-02,
                7.42029846e-02,
                1.84599757e00,
                1.98012757e00,
                -2.20554972e00,
                3.34040195e-01,
                2.00246739e00,
                2.29901099e00,
                1.93188477e00,
                -1.31648827e00,
                1.72804344e00,
                -6.20326877e-01,
                3.84516406e00,
            ],
            [
                -6.15004122e-01,
                1.92523181e00,
                3.10213834e-01,
                2.38560820e00,
                1.64980784e-01,
                2.47012734e00,
                2.28691030e00,
                2.13766146e00,
                3.09507060e00,
                2.20407224e00,
                1.40496099e00,
                2.09778953e00,
                1.54510307e00,
                2.78528047e00,
                9.87384498e-01,
                5.63294947e-01,
                8.49761486e-01,
                7.64072478e-01,
                5.16110849e00,
                2.50777125e-01,
                2.42723775e00,
                4.98759478e-01,
                8.36611807e-01,
                3.27741504e-02,
                6.42367423e-01,
                3.23579621e00,
                8.78783286e-01,
                2.65701032e00,
                -1.51014948e00,
                -4.03748602e-01,
                -8.78587902e-01,
                6.55948639e-01,
                -9.01640177e-01,
                1.27328143e-01,
                2.17064857e00,
                4.79705900e-01,
                -4.91857380e-01,
                2.11087108e00,
                3.72797132e00,
                1.34697601e-01,
                -7.48027623e-01,
                2.03303409e00,
                2.34893680e00,
                1.51575696e00,
                5.71517587e-01,
                5.51007152e-01,
                6.72422469e-01,
                1.51508331e00,
                2.80915260e00,
                2.19893193e00,
                3.63734394e-01,
                1.48130381e00,
                1.58301353e00,
                -4.83188033e-01,
                6.50487319e-02,
                1.48572290e00,
                -4.63951230e-02,
                -1.05817163e00,
                1.13302815e00,
                1.52109635e00,
                8.68799746e-01,
                -6.54618323e-01,
                2.29179120e00,
                9.32343721e-01,
            ],
            [
                -1.02231526e00,
                2.15637255e00,
                8.63004267e-01,
                1.49697900e00,
                3.77555400e-01,
                4.79371834e00,
                2.18187737e00,
                -4.08073694e-01,
                3.80424547e00,
                3.28488499e-01,
                2.91166544e-01,
                5.69210112e-01,
                -1.45953476e-01,
                4.92219508e-01,
                2.25897360e00,
                2.39856291e00,
                2.32339311e00,
                1.36280978e00,
                2.68326020e00,
                2.23726583e00,
                4.71672773e00,
                -2.41960555e-01,
                -6.41871452e-01,
                -1.37849063e-01,
                1.51697564e00,
                1.34768224e00,
                -6.36143982e-01,
                3.03063059e00,
                -6.52150035e-01,
                -6.94706917e-01,
                4.26955014e-01,
                1.81754500e-01,
                1.36751628e00,
                4.18924630e-01,
                2.72608232e00,
                -7.62602270e-01,
                -1.15675390e-01,
                1.55015421e00,
                2.05698538e00,
                3.58967841e-01,
                -3.46183181e00,
                1.90020382e00,
                3.29225719e-01,
                1.65795767e00,
                3.00538850e00,
                1.80392396e00,
                -6.38559103e-01,
                5.82976699e-01,
                3.54067326e00,
                2.16024518e00,
                7.44659007e-02,
                2.58961582e00,
                5.13008296e-01,
                3.76492906e00,
                1.21034771e-01,
                1.44272342e-01,
                2.16961533e-01,
                -1.14567757e-01,
                9.73347664e-01,
                6.59408927e-01,
                2.21385646e00,
                -1.04468942e00,
                1.08244288e00,
                1.17774665e00,
            ],
            [
                1.73727131e00,
                -1.29554939e00,
                1.31618273e00,
                1.05746877e00,
                1.09825246e-01,
                9.93288338e-01,
                2.78267789e00,
                2.36139941e00,
                3.23161030e00,
                2.45077252e-01,
                2.84062123e00,
                -1.14497554e00,
                1.25241888e00,
                -3.92041616e-02,
                1.18163025e00,
                3.32442379e00,
                -8.29961672e-02,
                2.94541621e00,
                4.43808824e-01,
                1.86333907e00,
                2.65909696e00,
                2.86817670e00,
                3.54500175e-01,
                4.52762604e00,
                3.93607903e00,
                4.39560843e00,
                6.52227402e-01,
                2.48165703e00,
                -6.97565079e-02,
                -1.57223475e00,
                -1.72357881e00,
                1.36772588e-01,
                1.08740823e-02,
                8.45155895e-01,
                2.43983362e-02,
                1.70513165e00,
                7.94518948e-01,
                4.30561781e-01,
                -1.20468460e-01,
                3.03318977e00,
                2.61915874e00,
                6.86616850e00,
                3.18797612e00,
                -9.57937002e-01,
                -3.89603353e00,
                2.85178638e00,
                -2.30784178e00,
                3.06549382e00,
                2.10986042e00,
                4.16084337e00,
                -1.04730546e00,
                2.26447776e-01,
                1.28120792e00,
                7.16031730e-01,
                1.44787443e00,
                3.48816895e00,
                2.38459992e00,
                2.62917209e00,
                -1.36497056e00,
                2.64265275e00,
                2.13323331e00,
                3.69102407e00,
                3.31435847e00,
                -2.82501485e-02,
            ],
            [
                9.21068549e-01,
                5.30967414e-02,
                -4.67774808e-01,
                6.76693320e-01,
                -1.31738186e00,
                1.75769305e00,
                4.49978209e00,
                8.63243461e-01,
                4.35214138e00,
                1.29374218e00,
                5.55299950e00,
                7.44466782e-01,
                2.94823194e00,
                -8.62419426e-01,
                5.88654518e-01,
                2.33071184e00,
                -1.38522089e00,
                2.20994377e00,
                8.56967211e-01,
                4.24994349e-01,
                1.07725000e00,
                4.56954002e00,
                -1.09210169e00,
                6.08523178e00,
                6.98644638e-01,
                5.94346046e-01,
                5.07137418e-01,
                2.76918411e00,
                7.02818096e-01,
                -1.16839194e00,
                -2.44228005e-01,
                -1.43439591e00,
                1.35015893e00,
                2.36504364e00,
                2.73242235e00,
                -1.73874509e00,
                -3.26049328e-03,
                1.40521073e00,
                6.76024556e-02,
                1.26458275e00,
                1.37417340e00,
                5.49824667e00,
                1.87933064e00,
                -1.44267690e00,
                -3.04753304e00,
                9.34975266e-01,
                -2.47016740e00,
                1.94030309e00,
                2.53315878e00,
                3.34717512e00,
                9.54726398e-01,
                -1.66520762e00,
                1.16518188e00,
                -3.57280076e-02,
                3.23623967e00,
                2.71055079e00,
                2.84211683e00,
                4.51837587e00,
                -3.51831317e-02,
                4.76910400e00,
                -1.35353982e00,
                3.37543607e00,
                4.38160276e00,
                1.87154245e00,
            ],
            [
                4.49691653e-01,
                -9.33344424e-01,
                -3.23174357e00,
                9.58482742e-01,
                -1.70231509e00,
                2.56687343e-01,
                3.55148649e00,
                1.95486283e00,
                -1.32599130e-01,
                1.78356934e00,
                3.83494473e00,
                2.27253914e00,
                1.25509501e-02,
                4.41403508e-01,
                2.54790783e00,
                -1.20292091e00,
                1.51880264e00,
                1.78743088e00,
                4.45846319e00,
                2.00633478e00,
                1.44756198e-01,
                1.08689213e00,
                -2.64912009e00,
                1.53960800e00,
                1.15162480e00,
                5.10639000e00,
                1.57132101e00,
                3.28193545e-01,
                -3.20204049e-02,
                5.15418828e-01,
                -4.69063222e-02,
                5.19496202e-01,
                -3.12976122e-01,
                4.43301582e00,
                -3.89653111e00,
                5.04955196e00,
                1.55091512e00,
                -1.69648695e00,
                -8.52413654e-01,
                -8.41709971e-01,
                2.42502093e00,
                4.17246675e00,
                1.65290976e00,
                -5.84828317e-01,
                -1.73806858e00,
                -3.75910580e-01,
                2.55819082e00,
                6.29656196e-01,
                1.21726751e00,
                1.78709912e00,
                -6.15739822e-01,
                8.73187900e-01,
                1.05297041e00,
                -3.91905129e-01,
                3.34197974e00,
                1.92704463e00,
                3.00968289e-01,
                4.55622435e-01,
                6.55164957e00,
                3.40055466e00,
                8.89938951e-01,
                1.75247741e00,
                2.07979488e00,
                2.28998041e00,
            ],
        ],
        dtype=np.float32,
    )

    gt_conf_prob = np.array(
        [
            [
                0.05811495,
                0.02137929,
                0.02137929,
                0.02137929,
                0.02137929,
                0.05811495,
                0.05811495,
                0.05811495,
                0.05811495,
                0.02137929,
                0.05811495,
                0.05811495,
                0.05811495,
                0.05811495,
                0.05811495,
                0.05811495,
                0.05811495,
                0.05811495,
                0.05811495,
                0.05811495,
                0.02137929,
            ],
            [
                0.05811495,
                0.02137929,
                0.02137929,
                0.02137929,
                0.02137929,
                0.05811495,
                0.05811495,
                0.05811495,
                0.05811495,
                0.02137929,
                0.05811495,
                0.05811495,
                0.05811495,
                0.05811495,
                0.05811495,
                0.05811495,
                0.05811495,
                0.05811495,
                0.05811495,
                0.05811495,
                0.02137929,
            ],
            [
                0.05811495,
                0.02137929,
                0.02137929,
                0.02137929,
                0.02137929,
                0.05811495,
                0.05811495,
                0.05811495,
                0.05811495,
                0.02137929,
                0.05811495,
                0.05811495,
                0.05811495,
                0.05811495,
                0.05811495,
                0.05811495,
                0.05811495,
                0.05811495,
                0.05811495,
                0.05811495,
                0.02137929,
            ],
            [
                0.05811495,
                0.02137929,
                0.02137929,
                0.02137929,
                0.02137929,
                0.05811495,
                0.05811495,
                0.05811495,
                0.05811495,
                0.02137929,
                0.05811495,
                0.05811495,
                0.05811495,
                0.05811495,
                0.05811495,
                0.05811495,
                0.05811495,
                0.05811495,
                0.05811495,
                0.05811495,
                0.02137929,
            ],
            [
                0.05811495,
                0.02137929,
                0.02137929,
                0.02137929,
                0.02137929,
                0.05811495,
                0.05811495,
                0.05811495,
                0.05811495,
                0.02137929,
                0.05811495,
                0.05811495,
                0.05811495,
                0.05811495,
                0.05811495,
                0.05811495,
                0.05811495,
                0.05811495,
                0.05811495,
                0.05811495,
                0.02137929,
            ],
            [
                0.05811495,
                0.02137929,
                0.02137929,
                0.02137929,
                0.02137929,
                0.05811495,
                0.05811495,
                0.05811495,
                0.05811495,
                0.02137929,
                0.05811495,
                0.05811495,
                0.05811495,
                0.05811495,
                0.05811495,
                0.05811495,
                0.05811495,
                0.05811495,
                0.05811495,
                0.05811495,
                0.02137929,
            ],
            [
                0.06812549,
                0.02506196,
                0.02506196,
                0.02506196,
                0.02506196,
                0.02506196,
                0.02506196,
                0.02506196,
                0.02506196,
                0.02506196,
                0.06812549,
                0.06812549,
                0.06812549,
                0.06812549,
                0.06812549,
                0.06812549,
                0.06812549,
                0.06812549,
                0.06812549,
                0.06812549,
                0.02506196,
            ],
        ],
        dtype=np.float32,
    )

    gt_probs = pd.DataFrame(
        data={
            "sample_id": [0, 0, 1, 2, 2, 4, 4],
            "split": ["inference"] * _num_pred_spans,
            "is_pred": [True] * _num_pred_spans,
            "span_start": [4, 12, 4, 0, 15, 0, 7],
            "span_end": [7, 14, 6, 3, 19, 2, 8],
            "pred": [
                "Questions About the Product",
                "Product Usage",
                "Renew",
                "Questions About the Product",
                "Potential Upsell",
                "Action item accomplished",
                "Action item accomplished",
            ],
            "inference_name": ["inf-name"] * _num_pred_spans,
        }
    )


NER_INFERENCE_DATA_TOKEN_INDICES = [
    [
        0,
        4,
        5,
        7,
        8,
        15,
        16,
        19,
        19,
        20,
        21,
        24,
        25,
        28,
        29,
        32,
        33,
        40,
        41,
        48,
        49,
        50,
        51,
        54,
        55,
        58,
        59,
        67,
        68,
        74,
        75,
        80,
        81,
        82,
    ],
    [0, 5, 6, 9, 10, 13, 14, 18, 19, 31, 32, 39],
    [
        0,
        3,
        4,
        7,
        8,
        15,
        16,
        20,
        21,
        28,
        29,
        32,
        33,
        35,
        36,
        39,
        40,
        45,
        45,
        46,
        46,
        51,
        52,
        55,
        56,
        59,
        60,
        67,
        68,
        72,
        73,
        80,
        81,
        84,
        85,
        86,
        86,
        92,
        92,
        93,
        93,
        98,
    ],
    [0, 1, 2, 6, 7, 11, 12, 19, 20, 23, 24, 27, 28, 35],
    [0, 4, 5, 8, 9, 11, 11, 12, 13, 17, 18, 25, 26, 28, 29, 36],
]
