/* Minimal AWNN API header — created from board symbols + detect_engine.c usage */
#ifndef AWNN_H
#define AWNN_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Awnn_Context Awnn_Context_t;

void            awnn_init(int memory_size);
Awnn_Context_t* awnn_create(char* nb_path);
void            awnn_set_input_buffers(Awnn_Context_t* ctx, void* input);
void            awnn_run(Awnn_Context_t* ctx);
float**         awnn_get_output_buffers(Awnn_Context_t* ctx);
void            awnn_destroy(Awnn_Context_t* ctx);
void            awnn_uninit(void);

#ifdef __cplusplus
}
#endif

#endif /* AWNN_H */
