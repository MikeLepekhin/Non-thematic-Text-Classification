## Estimation of the GPT-3 generators


10 epochs

| Lang | Model        | Prefix length | Genre accuracy | Topic accuracy |
|------|--------------|---------------|----------------|----------------|
| ru   | GPT-3 Small  | 10            | 0.393          | 1.0            |
| ru   | GPT-3 Small  | 50            | 0.364          | 0.97           |
| ru   | GPT-3 Medium | 10            | 0.357          | 0.99           |
| ru   | GPT-3 Medium | 50            | 0.305          | 0.97           |
| en   | GPT-3 Small  | 10            | 0.423          | 0.99           |
| en   | GPT-3 Small  | 50            | 0.346          | 0.94           |
| en   | GPT-3 Medium | 10            | 0.364          | 1.0            |
| en   | GPT-3 Medium | 50            | 0.327          | 0.95           |


30 epochs

| Lang | Model        | Prefix length | Genre accuracy | Topic accuracy |
|------|--------------|---------------|----------------|----------------|
| ru   | GPT-3 Small  | 10            | 0.426          |                |
| ru   | GPT-3 Medium | 10            | 0.423          |                |
| en   | GPT-3 Small  | 10            | 0.422          |                |
| en   | GPT-3 Medium | 10            | 0.42           |                |


Big train dataset (6 epochs)

| Lang | Model        | Prefix length | Genre accuracy | Topic accuracy |
|------|--------------|---------------|----------------|----------------|
| ru   | GPT-3 Small  | 10            | 0.266          |                |
| en   | GPT-3 Small  | 10            | 0.390          |                |