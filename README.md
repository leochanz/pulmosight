# PulmoSight

PulmoSight is a lung CT analysis platform with a web frontend, an application backend, and a separate model inference server.

## Deployment Overview

- **Frontend + Backend** are deployed on the same **Baidu AI Cloud (BCC) instance**.
- **Model Server** is deployed on a **separate BCC instance**.

## Public Access

- Website EIP (frontend/backend instance): **154.85.62.123**
- Model server EIP: **154.85.42.127**

You can access the website via:

- http://154.85.62.123
- https://pulmosight.top

`pulmosight.top` points to the same website as `http://154.85.62.123`.
