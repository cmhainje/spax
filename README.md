# spax

*hunting for axions in SPHEREx*

## authors

- Connor Hainje, NYU
- Elena Pinetti, CCA
- David W Hogg, NYU+

## data

This project makes use of publicly available SPHEREx Quick Release data, available at
[IRSA](https://irsa.ipac.caltech.edu/Missions/spherex.html)
or 
[AWS S3](https://registry.opendata.aws/spherex-qr/).
It also makes heavy use of the [NYU SPHEREx Talltable](https://connorhainje.com/talltable),
a very tall, SQL-queryable database consisting of all individual pixels from QR images.
The portion of the database used for this project is described in `talltable_paths.txt`.
Please note that this portion of the database weighs 18 TB!

For comparisons to existing limits, see [https://cajohare.github.io/AxionLimits/](https://cajohare.github.io/AxionLimits/).

## results

Our results are provided in `data/axion_limits.csv`.

## installation

You'll need both this repo *and* the talltable repo.

```
git clone https://github.com/cmhainje/talltable.git
cd talltable && uv sync && cd ..
git clone https://github.com/cmhainje/spax.git
cd spax && uv sync
```

