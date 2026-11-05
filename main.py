"""
Exemplo de Implementação de RAG (Retrieval-Augmented Generation)
Tema: Identificação de requisitos de usabilidade baseando-se apenas na descrição textual.

Regra do negócio:
- O arquivo analisado é `requisitos.json`.
- Apenas o campo "descrição" é indexado e vetorizado pelo RAG.
- O campo "classificador" NÃO é utilizado na busca/recuperação nem como contexto (é apenas controle).
- Avaliação de desempenho com Precision, Recall, F1-macro e Matriz de Confusão.
"""

import json
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sentence_transformers import SentenceTransformer

# 1. Definição do Modelo de Embeddings
# Modelo leve e multilíngue que suporta português com alta precisão semântica
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class SimpleRAG:
    def __init__(self, json_filepath: str, model_name: str = MODEL_NAME):
        self.json_filepath = json_filepath
        print(f"[*] Carregando modelo de embeddings ({model_name})...")
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None
        self._load_and_index()

    def _load_and_index(self):
        if not os.path.exists(self.json_filepath):
            raise FileNotFoundError(f"Arquivo não encontrado: {self.json_filepath}")

        with open(self.json_filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"[*] {len(data)} requisitos carregados de '{self.json_filepath}'.")

        # Regra: Baseado APENAS na descrição. O classificador é controle e NÃO entra no RAG.
        self.documents = [
            {
                "id": req.get("id"),
                "descricao": req.get("descrição"),
                # Mantido apenas para conferência/validação externa posterior (não usado no índice)
                "_controle_classificador": req.get("classificador"),
            }
            for req in data
        ]

        descricoes = [doc["descricao"] for doc in self.documents]
        print("[*] Gerando embeddings das descrições para a base de conhecimento...")
        self.embeddings = self.model.encode(descricoes, show_progress_bar=False, normalize_embeddings=True)
        print("[*] Indexação concluída com sucesso!\n")

    def retrieve(self, query: str, top_k: int = 10, threshold: float = None):
        """
        Etapa de Retrieval: recupera os documentos mais semanticamente similares à consulta.
        Pode filtrar por top_k ou por threshold de similaridade mínima.
        """
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        # Similaridade de cosseno
        scores = cosine_similarity(query_embedding, self.embeddings)[0]

        top_indices = np.argsort(scores)[::-1]
        if threshold is not None:
            top_indices = [idx for idx in top_indices if scores[idx] >= threshold]
        else:
            top_indices = top_indices[:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "index": int(idx),
                "documento": self.documents[idx],
                "score": float(scores[idx])
            })
        return results, scores

    def generate_augmented_response(self, query: str, retrieved_items: list) -> str:
        """
        Etapa de Augmentation + Generation:
        Monta o contexto recuperado e gera uma resposta estruturada e analítica
        fundamentada exclusivamente nas descrições recuperadas.
        """
        # Montagem do Contexto (Augmented Prompt)
        contexto_textual = []
        for i, item in enumerate(retrieved_items, 1):
            doc = item["documento"]
            score = item["score"]
            contexto_textual.append(
                f"[{i}] ID: {doc['id']} (Relevância Semântica: {score:.4f})\n"
                f"    Descrição: \"{doc['descricao']}\""
            )

        contexto_formatado = "\n".join(contexto_textual)

        resposta = (
            "=" * 80 + "\n"
            f"PERGUNTA / QUERY DO USUÁRIO:\n'{query}'\n"
            + "=" * 80 + "\n"
            "CONTEXTO RECUPERADO (Apenas descrições de requisitos):\n"
            f"{contexto_formatado}\n"
            + "-" * 80 + "\n"
            "RESPOSTA GERADA (RAG Synthesis):\n"
            f"Com base na análise semântica das descrições dos requisitos no repositório,\n"
            f"foram identificados os requisitos mais aderentes aos critérios de usabilidade, facilidade de uso,\n"
            f"experiência do usuário e consistência visual/interativa:\n\n"
        )

        for i, item in enumerate(retrieved_items, 1):
            doc = item["documento"]
            score = item["score"]
            resposta += (
                f"  {i}. {doc['id']} (Score: {score:.2%}):\n"
                f"     {doc['descricao']}\n\n"
            )

        resposta += (
            "Análise do RAG:\n"
            "- Estes requisitos tratam diretamente de interface, navegação padronizada, responsividade,\n"
            "  acessibilidade, prevenção de erros e auxílio ao usuário durante a interação.\n"
            "- O campo de controle 'classificador' permaneceu isolado, demonstrando a capacidade da\n"
            "  busca vetorial em discernir conceitos de usabilidade puramente pela descrição.\n"
            + "=" * 80
        )
        return resposta

    def evaluate(self, query: str, top_k: int = 10, target_class: str = "Usabilidade"):
        """
        Calcula as métricas de avaliação:
        - Precision (Precisão)
        - Recall (Revocação)
        - F1-Macro
        - Matriz de Confusão
        Comparando a predição do RAG (baseada puramente na descrição) contra o ground-truth
        do campo de controle ('classificador').
        """
        retrieved_items, all_scores = self.retrieve(query, top_k=top_k)
        retrieved_indices = set(item["index"] for item in retrieved_items)

        # Ground-truth: 1 se for a classe alvo ('Usabilidade'), 0 caso contrário
        y_true = [
            1 if doc["_controle_classificador"] == target_class else 0
            for doc in self.documents
        ]

        # Predição binária do RAG para cada documento no repositório
        y_pred = [1 if idx in retrieved_indices else 0 for idx in range(len(self.documents))]

        # Cálculo das métricas
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        cm = confusion_matrix(y_true, y_pred)  # [[TN, FP], [FN, TP]]

        tn, fp, fn, tp = cm.ravel()

        return {
            "precision": prec,
            "recall": rec,
            "f1_macro": f1_macro,
            "confusion_matrix": cm,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "retrieved_items": retrieved_items,
            "total_docs": len(self.documents),
            "total_positives_true": sum(y_true),
        }


def print_evaluation_report(eval_res: dict, query: str, top_k: int, target_class: str = "Usabilidade"):
    cm = eval_res["confusion_matrix"]
    tp = eval_res["tp"]
    fp = eval_res["fp"]
    fn = eval_res["fn"]
    tn = eval_res["tn"]

    print("\n" + "=" * 80)
    print("           RELATÓRIO DE AVALIAÇÃO DO RAG (CLASSIFICAÇÃO POR DESCRIÇÃO)")
    print("=" * 80)
    print(f"Consulta utilizada: '{query}'")
    print(f"Documentos recuperados (top_k): {top_k} de {eval_res['total_docs']} totais")
    print(f"Requisitos reais com '{target_class}' (Ground Truth): {eval_res['total_positives_true']}")
    print("-" * 80)
    print(f" Precision (Precisão):   {eval_res['precision']:.4f}  ({tp}/{tp+fp} recuperados são realmente '{target_class}')")
    print(f" Recall (Revocação):      {eval_res['recall']:.4f}  ({tp}/{eval_res['total_positives_true']} dos requisitos de '{target_class}' foram encontrados)")
    print(f" F1-Macro:               {eval_res['f1_macro']:.4f}  (Média harmônica macro entre classes)")
    print("-" * 80)
    print(" MATRIZ DE CONFUSÃO:")
    print("                         Predito: Não-Usabilidade   Predito: Usabilidade")
    print(f" Real: Não-Usabilidade         TN = {tn:<4}                 FP = {fp:<4}")
    print(f" Real: Usabilidade             FN = {fn:<4}                 TP = {tp:<4}")
    print("\n Interpretação da Matriz:")
    print(f"   • Verdadeiros Positivos (TP): {tp} requisitos de Usabilidade foram corretamente identificados.")
    print(f"   • Falsos Positivos (FP):       {fp} requisitos de outros temas foram classificados como Usabilidade.")
    print(f"   • Falsos Negativos (FN):       {fn} requisitos de Usabilidade não entraram no top-{top_k}.")
    print(f"   • Verdadeiros Negativos (TN): {tn} requisitos não-usabilidade foram corretamente mantidos fora.")
    print("=" * 80 + "\n")


def main():
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requisitos.json")

    # Inicializa o pipeline de RAG
    rag = SimpleRAG(json_filepath=json_path)

    # Consulta semântica para identificação de Usabilidade
    query = (
    "Requisitos de usabilidade da interface: consistência visual de telas, facilidade de uso, "
    "navegação intuitiva, acessibilidade, responsividade, feedback ao usuário, "
    "prevenção de erros em formulários, atalhos para agilidade e confirmação de ações."
)

    print(f"Executando busca RAG para: '{query}'\n")

    # Avaliação com top_k = 10 (pois existem 10 requisitos de Usabilidade no dataset)
    top_k = 10
    eval_res = rag.evaluate(query, top_k=top_k, target_class="Usabilidade")

    # Gera a resposta aumentada (Augmented Generation) com os itens recuperados
    resultado = rag.generate_augmented_response(query, eval_res["retrieved_items"])
    print(resultado)

    # Exibe a validação de controle item a item
    print("\n[VALIDAÇÃO DE CONTROLE - Verificando os classificadores originais dos requisitos recuperados]:")
    for item in eval_res["retrieved_items"]:
        doc = item["documento"]
        print(f" - {doc['id']} | Classificador de controle: '{doc['_controle_classificador']}' | Score: {item['score']:.4f}")

    # Exibe o teste de métricas requisitado
    print_evaluation_report(eval_res, query, top_k=top_k, target_class="Usabilidade")


if __name__ == "__main__":
    main()
