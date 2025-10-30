# GPURemBG

Removedor de fundo acelerado por GPU pensado para placas NVIDIA (ex: RTX 3060).  
O projeto expõe vários algoritmos de matting de última geração, cada um executando **exclusivamente na GPU** e, quando possível, utilizando FP16/AMP para explorar Tensor Cores.

## Principais recursos
- ✅ Modelos prontos de alta qualidade: `u2net`, `u2netp`, `isnet-general-use`, `rmbg-1.4`.
- ✅ Execução CUDA garantida (PyTorch + onnxruntime-gpu).
- ✅ FP16 / Tensor Cores habilitados por padrão nos modelos PyTorch.
- ✅ CLI única para varrer diretórios, gerar PNGs com transparência e medir o tempo de processamento.
- ✅ Download automático dos pesos (cache em `~/.cache/gpurembg`).

## Requisitos
- Linux Mint (ou similar) com driver NVIDIA + CUDA configurados.
- Python 3.9+ (recomendado 3.10/3.11).
- Placa NVIDIA com suporte a CUDA (ex: RTX 3060).

> **Verifique o CUDA antes de prosseguir**
```bash
nvidia-smi
```

## Instalação
Crie e ative um ambiente virtual Python:
```bash
python -m venv .venv
source .venv/bin/activate
```

Instale PyTorch **com suporte CUDA** (ajuste a versão de acordo com seu driver/CUDA):
```bash
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Depois instale as dependências do projeto:
```bash
pip install -r requirements.txt
pip install -e .
```

## Uso – processamento em lote
Organize suas imagens em uma pasta, por exemplo `data/input/`.

Execute o pipeline:
```bash
python -m gpurembg.cli \
  --input-dir data/input \
  --output-dir data/output \
  --models isnet rmbg14 \
  --device cuda:0 \
  --overwrite
```

### Parâmetros úteis
- `--models`: escolha uma ou mais opções dentre `u2net`, `u2netp`, `u2net-portrait`, `u2net-human`, `isnet`, `rmbg14`. Por padrão usamos `isnet` e `rmbg14` com threshold automático de 0.6.
- `--no-fp16`: desativa FP16 (útil para debug).
- `--alpha-threshold 0.9`: aplica um corte duro na máscara.
- `--refine-dilate 2`: adiciona dilatações 3x3 para recuperar contornos perdidos.
- `--refine-feather 3`: suaviza bordas com blur gaussiano (usa raio em pixels).
- `--json report.json`: salva estatísticas (número de imagens, tempo total e médio por modelo).

Os resultados são salvos em subpastas (uma por modelo) dentro do diretório de saída, sempre em PNG RGBA.

### Pesos pré-baixados (opcional)
Caso prefira baixar os pesos manualmente (por exemplo, direto do Google Drive):

1. Baixe os arquivos `*.pth` e `*.onnx`.
2. Coloque-os em `~/.cache/gpurembg/` (padrão) ou em qualquer pasta apontada por `--weights-dir`.
3. Use exatamente estes nomes:
   - `u2net.pth`
   - `u2netp.pth`
   - `u2net_portrait.pth`
   - `u2net_human_seg.pth`
   - `isnet-general-use.onnx`
   - `rmbg-1.4.onnx` (pode ser baixado em https://huggingface.co/briaai/RMBG-1.4 )

O pipeline detecta automaticamente os arquivos existentes e pula o download.

## Algoritmos incluídos
| Nome      | Framework       | Qualidade | Desempenho | Observações |
|-----------|-----------------|-----------|------------|-------------|
| `u2net`   | PyTorch (FP32)  | Alta      | Médio      | Versão base, agressiva em áreas complexas. |
| `u2netp`  | PyTorch (FP32)  | Boa       | Alto       | Versão compacta; ideal para lotes grandes. |
| `u2net-portrait` | PyTorch (FP32) | Alta | Médio | Treinado para retratos; preserva cabelo. |
| `u2net-human` | PyTorch (FP32) | Alta | Médio | Segmentação corporal completa. |
| `isnet`   | ONNX Runtime    | Altíssima | Alto       | IS-Net general-use; opera em 1024px. |
| `rmbg14`  | ONNX Runtime    | Alta      | Muito alto | Novo modelo rápido do projeto rembg. |

Todos os modelos baixam pesos automaticamente para `~/.cache/gpurembg`.  
Os backends ONNX usam `onnxruntime-gpu` e executam diretamente na CUDA.

## Validando o uso de GPU
- **PyTorch:** o código falha se `torch.cuda.is_available()` for `False`.
- **onnxruntime-gpu:** a execução lança exceção se a `CUDAExecutionProvider` não for inicializada.
- Monitore durante o processamento:
  ```bash
  watch -n 1 nvidia-smi
  ```

## Boas práticas de performance
1. Mantenha os drivers e o CUDA Toolkit atualizados.
2. Garanta que o ambiente está usando PyTorch + onnxruntime compilados para a mesma versão de CUDA.
3. Use SSD para leitura/escrita rápida em lotes grandes.
4. Ajuste `--alpha-threshold` apenas se precisar de máscaras mais duras; valores baixos podem introduzir ruído.

## Próximos passos sugeridos
- Converter os modelos ONNX para TensorRT (onnxruntime já utiliza kernels acelerados, mas TensorRT pode oferecer ganhos extras).
- Adicionar suporte a processamento assíncrono com múltiplos streams CUDA para uso intensivo em lotes.
- Integrar um visualizador/inspector com comparação lado a lado dos resultados entre modelos.

## Licença
Os modelos mantêm as licenças originais (U²-Net, IS-Net, RMBG). O código deste projeto é fornecido “como está”.
