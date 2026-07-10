# Dataset Sources and Attribution

This repository uses two public datasets for bounded external validation. The
repository does not redistribute the original image archives. The versioned
feature cache and summary outputs are derived artifacts for the stated
evaluation protocol.

## Laboro Tomato

- Official dataset name: Laboro Tomato: Instance segmentation dataset.
- Official repository: <https://github.com/laboroai/LaboroTomato>
- Official release page: <https://laboro.ai/activity/column/engineer/laboro-tomato/>
- Publisher: Laboro.AI, 2020.
- Original task: tomato object detection and instance segmentation across
  size and ripening-stage categories.
- Use in this repository: bounded external validation with 9,430 derived crop
  features, mapped to `mature`, `turning`, and `immature` evaluation labels.

### Recommended citation

```bibtex
@dataset{laboro_tomato_2020,
  author    = {{Laboro.AI}},
  title     = {Laboro Tomato: Instance Segmentation Dataset},
  year      = {2020},
  publisher = {Laboro.AI},
  url       = {https://github.com/laboroai/LaboroTomato}
}
```

The official release page describes the dataset under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), while the
current official repository README states
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Users
must follow the license attached to the exact source version they download.
This project does not grant any additional rights to the original images.

## Strawberry-DS

- Official dataset name: Strawberry-DS: Dataset of annotated strawberry fruits
  images with various developmental stages.
- Dataset record: <https://data.mendeley.com/datasets/z6dtfdpzz8/1>
- Dataset DOI: `10.17632/z6dtfdpzz8.1`.
- Dataset record: Mendeley Data, Version 1, published 2022.
- Use in this repository: external-domain transfer, recalibration sensitivity,
  and in-domain reference evaluation using 1,083 derived single-fruit crops.

### Recommended paper citation

```bibtex
@article{elhariri2023strawberryds,
  author  = {Elhariri, Esraa and El-Bendary, Nashwa and Saleh, Samir Mahmoud},
  title   = {Strawberry-DS: Dataset of annotated strawberry fruits images with various developmental stages},
  journal = {Data in Brief},
  volume  = {48},
  pages   = {109165},
  year    = {2023},
  doi     = {10.1016/j.dib.2023.109165},
  url     = {https://doi.org/10.1016/j.dib.2023.109165}
}
```

### Recommended data-record citation

```bibtex
@dataset{elbendary2022strawberryds,
  author    = {El-Bendary, Nashwa and Elhariri, Esraa},
  title     = {Strawberry-DS},
  year      = {2022},
  publisher = {Mendeley Data},
  version   = {1},
  doi       = {10.17632/z6dtfdpzz8.1},
  url       = {https://data.mendeley.com/datasets/z6dtfdpzz8/1}
}
```

The Mendeley Data record states
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The original publication describes
247 source RGB images; the 1,083 items reported by this repository are derived
single-fruit crops and must not be presented as the original image count.

## Attribution requirement

Any publication, release, or derivative benchmark using the external results
must cite the relevant dataset source above and retain the applicable license
and attribution terms. The dataset names and source records are not claims of
ownership by TAP-Correct.
