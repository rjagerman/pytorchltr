/*
 * This is a reasonably fast implementation to parse SVMrank files.
 * 
 * This implementation uses a DFA and performs a single pass over the input.
 * A second pass over in-memory data is performed to construct a dense feature
 * matrix.
 * 
 * This parser only supports ASCII-encoded SVMrank files using the format as
 * described in http://www.cs.cornell.edu/people/tj/svm_light/svm_rank.html.
 */
#ifndef _SVMRANK_PARSER_H
#define _SVMRANK_PARSER_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

// Constants
size_t BUFFER_SIZE = 8192;
int PARSE_OK = 0;
int PARSE_FILE_ERROR = 1;
int PARSE_FORMAT_ERROR = 2;
int PARSE_MEMORY_ERROR = 3;

// Parser DFA states
typedef enum {
    INVALID = 0,
    START_Y = 1,
    PROCESS_Y = 2,
    START_QID = 3,
    START_QID_Q = 4,
    START_QID_I = 5,
    START_QID_D = 6,
    START_QID_COLON = 7,
    PROCESS_QID = 8,
    START_FEAT_COL = 9,
    PROCESS_FEAT_COL = 10,
    START_FEAT_VAL_1 = 11,
    PROCESS_FEAT_VAL_1 = 12,
    START_FEAT_VAL_2 = 13,
    PROCESS_FEAT_VAL_2 = 14,
    START_FEAT_VAL_3 = 15,
    PROCESS_FEAT_VAL_3 = 16,
    SKIP = 17
} state;

// Parser DFA actions
typedef enum {
    RESET,
    PREPARE_Y,
    UPDATE_Y,
    STORE_Y,
    PREPARE_QID,
    UPDATE_QID,
    STORE_QID,
    PREPARE_FEAT_COL,
    UPDATE_FEAT_COL,
    STORE_FEAT_COL,
    PREPARE_FEAT_VAL,
    SET_FEAT_VAL_NEGATIVE,
    SET_FEAT_VAL_EXP_NEGATIVE,
    UPDATE_FEAT_VAL_1,
    UPDATE_FEAT_VAL_2,
    UPDATE_FEAT_VAL_3,
    STORE_FEAT_VAL,
} action;

// Matrix shape.
typedef struct shape {
    size_t rows;
    size_t cols;
} shape;

// DFA transition and action tables.
unsigned char TRANSITIONS[32][256];
unsigned char ACTIONS[32][256];

// Initializes the DFA transition table.
void init_transition_table() {
    TRANSITIONS[START_Y]['#'] = SKIP;
    TRANSITIONS[START_Y][' '] = START_Y;
    for (unsigned char c='0'; c<='9'; c++) { TRANSITIONS[START_Y][c] = PROCESS_Y; }
    for (unsigned char c='0'; c<='9'; c++) { TRANSITIONS[PROCESS_Y][c] = PROCESS_Y; }
    TRANSITIONS[PROCESS_Y][' '] = START_QID;
    TRANSITIONS[START_QID][' '] = START_QID;
    TRANSITIONS[START_QID]['q'] = START_QID_Q;
    TRANSITIONS[START_QID_Q]['i'] = START_QID_I;
    TRANSITIONS[START_QID_I]['d'] = START_QID_D;
    TRANSITIONS[START_QID_D][':'] = START_QID_COLON;
    for (unsigned char c='0'; c<='9'; c++) { TRANSITIONS[START_QID_COLON][c] = PROCESS_QID; }
    for (unsigned char c='0'; c<='9'; c++) { TRANSITIONS[PROCESS_QID][c] = PROCESS_QID; }
    TRANSITIONS[PROCESS_QID][' '] = START_FEAT_COL;
    TRANSITIONS[PROCESS_QID]['#'] = SKIP;
    TRANSITIONS[PROCESS_QID]['\r'] = SKIP;
    TRANSITIONS[PROCESS_QID]['\n'] = START_Y;
    TRANSITIONS[START_FEAT_COL][' '] = START_FEAT_COL;
    TRANSITIONS[START_FEAT_COL]['#'] = SKIP;
    TRANSITIONS[START_FEAT_COL]['\r'] = SKIP;
    TRANSITIONS[START_FEAT_COL]['\n'] = START_Y;
    for (unsigned char c='0'; c<='9'; c++) { TRANSITIONS[START_FEAT_COL][c] = PROCESS_FEAT_COL; }
    for (unsigned char c='0'; c<='9'; c++) { TRANSITIONS[PROCESS_FEAT_COL][c] = PROCESS_FEAT_COL; }
    TRANSITIONS[PROCESS_FEAT_COL][':'] = START_FEAT_VAL_1;
    TRANSITIONS[START_FEAT_VAL_1]['-'] = START_FEAT_VAL_1;
    for (unsigned char c='0'; c<='9'; c++) { TRANSITIONS[START_FEAT_VAL_1][c] = PROCESS_FEAT_VAL_1; }
    for (unsigned char c='0'; c<='9'; c++) { TRANSITIONS[PROCESS_FEAT_VAL_1][c] = PROCESS_FEAT_VAL_1; }
    TRANSITIONS[PROCESS_FEAT_VAL_2]['e'] = START_FEAT_VAL_3;
    TRANSITIONS[PROCESS_FEAT_VAL_2]['E'] = START_FEAT_VAL_3;
    TRANSITIONS[PROCESS_FEAT_VAL_1][' '] = START_FEAT_COL;
    TRANSITIONS[PROCESS_FEAT_VAL_1]['#'] = SKIP;
    TRANSITIONS[PROCESS_FEAT_VAL_1]['\r'] = SKIP;
    TRANSITIONS[PROCESS_FEAT_VAL_1]['\n'] = START_Y;
    TRANSITIONS[PROCESS_FEAT_VAL_1]['.'] = START_FEAT_VAL_2;
    for (unsigned char c='0'; c<='9'; c++) { TRANSITIONS[START_FEAT_VAL_2][c] = PROCESS_FEAT_VAL_2; }
    for (unsigned char c='0'; c<='9'; c++) { TRANSITIONS[PROCESS_FEAT_VAL_2][c] = PROCESS_FEAT_VAL_2; }
    TRANSITIONS[PROCESS_FEAT_VAL_2]['e'] = START_FEAT_VAL_3;
    TRANSITIONS[PROCESS_FEAT_VAL_2]['E'] = START_FEAT_VAL_3;
    TRANSITIONS[PROCESS_FEAT_VAL_2][' '] = START_FEAT_COL;
    TRANSITIONS[PROCESS_FEAT_VAL_2]['#'] = SKIP;
    TRANSITIONS[PROCESS_FEAT_VAL_2]['\r'] = SKIP;
    TRANSITIONS[PROCESS_FEAT_VAL_2]['\n'] = START_Y;
    TRANSITIONS[START_FEAT_VAL_3]['-'] = PROCESS_FEAT_VAL_3;
    TRANSITIONS[START_FEAT_VAL_3]['+'] = PROCESS_FEAT_VAL_3;
    for (unsigned char c='0'; c<='9'; c++) { TRANSITIONS[START_FEAT_VAL_3][c] = PROCESS_FEAT_VAL_3; }
    for (unsigned char c='0'; c<='9'; c++) { TRANSITIONS[PROCESS_FEAT_VAL_3][c] = PROCESS_FEAT_VAL_3; }
    TRANSITIONS[PROCESS_FEAT_VAL_3][' '] = START_FEAT_COL;
    TRANSITIONS[PROCESS_FEAT_VAL_3]['#'] = SKIP;
    TRANSITIONS[PROCESS_FEAT_VAL_3]['\r'] = SKIP;
    TRANSITIONS[PROCESS_FEAT_VAL_3]['\n'] = START_Y;
    for (size_t c=0; c<256; c++) { TRANSITIONS[SKIP][c] = SKIP; }
    TRANSITIONS[SKIP]['\n'] = START_Y;
}

// Initializes the DFA action table.
void init_action_table() {
    for (unsigned char c='0'; c<='9'; c++) { ACTIONS[START_Y][c] = PREPARE_Y; }
    for (unsigned char c='0'; c<='9'; c++) { ACTIONS[PROCESS_Y][c] = UPDATE_Y; }
    ACTIONS[PROCESS_Y][' '] = STORE_Y;
    for (unsigned char c='0'; c<='9'; c++) { ACTIONS[START_QID_COLON][c] = PREPARE_QID; }
    for (unsigned char c='0'; c<='9'; c++) { ACTIONS[PROCESS_QID][c] = UPDATE_QID; }
    ACTIONS[PROCESS_QID][' '] = STORE_QID;
    for (unsigned char c='0'; c<='9'; c++) { ACTIONS[START_FEAT_COL][c] = PREPARE_FEAT_COL; }
    for (unsigned char c='0'; c<='9'; c++) { ACTIONS[PROCESS_FEAT_COL][c] = UPDATE_FEAT_COL; }
    ACTIONS[PROCESS_FEAT_COL][':'] = STORE_FEAT_COL;
    ACTIONS[START_FEAT_VAL_1]['-'] = SET_FEAT_VAL_NEGATIVE;
    for (unsigned char c='0'; c<='9'; c++) { ACTIONS[START_FEAT_VAL_1][c] = PREPARE_FEAT_VAL; }
    for (unsigned char c='0'; c<='9'; c++) { ACTIONS[PROCESS_FEAT_VAL_1][c] = UPDATE_FEAT_VAL_1; }
    for (unsigned char c='0'; c<='9'; c++) { ACTIONS[START_FEAT_VAL_2][c] = UPDATE_FEAT_VAL_2; }
    for (unsigned char c='0'; c<='9'; c++) { ACTIONS[PROCESS_FEAT_VAL_2][c] = UPDATE_FEAT_VAL_2; }
    ACTIONS[START_FEAT_VAL_3]['-'] = SET_FEAT_VAL_EXP_NEGATIVE;
    for (unsigned char c='0'; c<='9'; c++) { ACTIONS[START_FEAT_VAL_3][c] = UPDATE_FEAT_VAL_3; }
    for (unsigned char c='0'; c<='9'; c++) { ACTIONS[PROCESS_FEAT_VAL_3][c] = UPDATE_FEAT_VAL_3; }
    ACTIONS[PROCESS_FEAT_VAL_1][' '] = STORE_FEAT_VAL;
    ACTIONS[PROCESS_FEAT_VAL_1]['#'] = STORE_FEAT_VAL;
    ACTIONS[PROCESS_FEAT_VAL_1]['\r'] = STORE_FEAT_VAL;
    ACTIONS[PROCESS_FEAT_VAL_1]['\n'] = STORE_FEAT_VAL;
    ACTIONS[PROCESS_FEAT_VAL_2][' '] = STORE_FEAT_VAL;
    ACTIONS[PROCESS_FEAT_VAL_2]['#'] = STORE_FEAT_VAL;
    ACTIONS[PROCESS_FEAT_VAL_2]['\r'] = STORE_FEAT_VAL;
    ACTIONS[PROCESS_FEAT_VAL_2]['\n'] = STORE_FEAT_VAL;
    ACTIONS[PROCESS_FEAT_VAL_3][' '] = STORE_FEAT_VAL;
    ACTIONS[PROCESS_FEAT_VAL_3]['#'] = STORE_FEAT_VAL;
    ACTIONS[PROCESS_FEAT_VAL_3]['\r'] = STORE_FEAT_VAL;
    ACTIONS[PROCESS_FEAT_VAL_3]['\n'] = STORE_FEAT_VAL;
}

// Init function
void init_svmrank_parser() {
    init_transition_table();
    init_action_table();
}

// Main SVMrank parse function.
int parse_svmrank_file(char* path, double** xs_out, shape* xs_shape, int** ys_out, long** qids_out) {

    // Main file reading variables.
    char buffer[BUFFER_SIZE];
    size_t bytes_read = 0;
    long total_read = 0;
    FILE* fp = fopen(path, "r");

    // If we cannot open the file for reading, return an appropriate error code.
    if (fp == NULL) {
        return PARSE_FILE_ERROR;
    }

    // Initialize DFA variables
    action current_action = 0;
    state current_state = START_Y;

    // Current parse variables.
    int y = 0;
    long qid = 0;
    unsigned long row = 0;
    unsigned int col = 0;
    unsigned long nr_cols = 0;
    unsigned long min_col = 0;
    int set_min_col = 1;
    long decplaces = 0;
    long sign = 1;
    long val = 0;
    long expval = 0;
    long expsign = 1;
    double feat_val = 0.0;

    // Allocate output data holders
    size_t ys_capacity = 100;
    size_t ys_cursor = 0;
    int* ys = malloc(ys_capacity * sizeof(int));
    if (ys == NULL) {
        return PARSE_MEMORY_ERROR;
    }
    int* realloc_ys;
    size_t qids_capacity = 100;
    size_t qids_cursor = 0;
    long* qids = malloc(qids_capacity * sizeof(long));
    if (qids == NULL) {
        free(ys);
        return PARSE_MEMORY_ERROR;
    }
    long* realloc_qids;
    size_t rows_capacity = 1000;
    size_t rows_cursor = 0;
    long* rows = malloc(rows_capacity * sizeof(long));
    if (rows == NULL) {
        free(ys);
        free(qids);
        return PARSE_MEMORY_ERROR;
    }
    long* realloc_rows;
    size_t cols_capacity = 1000;
    size_t cols_cursor = 0;
    int* cols = malloc(cols_capacity * sizeof(int));
    if (cols == NULL) {
        free(ys);
        free(qids);
        free(rows);
        return PARSE_MEMORY_ERROR;
    }
    int* realloc_cols;
    size_t vals_capacity = 1000;
    size_t vals_cursor = 0;
    double* vals = malloc(vals_capacity * sizeof(double));
    if (vals == NULL) {
        free(ys);
        free(qids);
        free(rows);
        free(cols);
        return PARSE_MEMORY_ERROR;
    }
    double* realloc_vals;

    // Read file in buffer-sized chunks and parse them.
    do {
        bytes_read = fread(buffer, sizeof(char), BUFFER_SIZE, fp);
        total_read += bytes_read;

        // Iterate each character in the current buffer.
        for (size_t i=0; i<bytes_read; i++) {
            unsigned char c = buffer[i];

            // Get the DFA action to perform given the current state and char.
            current_action = ACTIONS[current_state][c];

            // Execute the DFA action.
            switch (current_action) {
                case RESET:
                    break;
                case PREPARE_Y:
                    row += 1;
                    y = c - '0';
                    break;
                case UPDATE_Y:
                    y = y * 10 + (c - '0');
                    break;
                case STORE_Y:
                    if (ys_cursor >= ys_capacity) {
                        ys_capacity = 1 + (ys_cursor * 3 / 2);
                        realloc_ys = realloc(
                            ys, ys_capacity * sizeof(int));
                        if (realloc_ys == NULL) {
                            free(ys);
                            free(qids);
                            free(rows);
                            free(cols);
                            free(vals);
                            return PARSE_MEMORY_ERROR;
                        }
                        ys = realloc_ys;
                    }
                    ys[ys_cursor] = y;
                    ys_cursor += 1;
                    break;
                case PREPARE_QID:
                    qid = c - '0';
                    break;
                case UPDATE_QID:
                    qid = qid * 10 + (c - '0');
                    break;
                case STORE_QID:
                    if (qids_cursor >= qids_capacity) {
                        qids_capacity = 1 + (qids_cursor * 3 / 2);
                        realloc_qids = realloc(
                            qids, qids_capacity * sizeof(long));
                        if (realloc_qids == NULL) {
                            free(ys);
                            free(qids);
                            free(rows);
                            free(cols);
                            free(vals);
                            return PARSE_MEMORY_ERROR;
                        }
                        qids = realloc_qids;
                    }
                    qids[qids_cursor] = qid;
                    qids_cursor += 1;
                    break;
                case PREPARE_FEAT_COL:
                    col = c - '0';
                    sign = 1;
                    decplaces = 0;
                    expsign = 1;
                    expval = 0;
                    break;
                case UPDATE_FEAT_COL:
                    col = col * 10 + (c - '0');
                    break;
                case STORE_FEAT_COL:
                    if (rows_cursor >= rows_capacity) {
                        rows_capacity = 1 + (rows_cursor * 3 / 2);
                        realloc_rows = realloc(
                            rows, rows_capacity * sizeof(long));
                        if (realloc_rows == NULL) {
                            free(ys);
                            free(qids);
                            free(rows);
                            free(cols);
                            free(vals);
                            return PARSE_MEMORY_ERROR;
                        }
                        rows = realloc_rows;
                    }
                    rows[rows_cursor] = row - 1;
                    rows_cursor += 1;
                    if (cols_cursor >= cols_capacity) {
                        cols_capacity = 1 + (cols_cursor * 3 / 2);
                        realloc_cols = realloc(
                            cols, cols_capacity * sizeof(int));
                        if (realloc_cols == NULL) {
                            free(ys);
                            free(qids);
                            free(rows);
                            free(cols);
                            free(vals);
                            return PARSE_MEMORY_ERROR;
                        }
                        cols = realloc_cols;
                    }
                    cols[cols_cursor] = col;
                    cols_cursor += 1;

                    if (set_min_col == 1 || col < min_col) {
                        min_col = col;
                        set_min_col = 0;
                    }
                    if (col + 1 > nr_cols) {
                        nr_cols = col + 1;
                    }
                    break;
                case SET_FEAT_VAL_NEGATIVE:
                    sign = -1;
                    break;
                case PREPARE_FEAT_VAL:
                    val = c - '0';
                    break;
                case UPDATE_FEAT_VAL_2:
                    decplaces += 1;
                case UPDATE_FEAT_VAL_1:
                    val = val * 10 + (c - '0');
                    break;
                case SET_FEAT_VAL_EXP_NEGATIVE:
                    expsign = -1;
                    break;
                case UPDATE_FEAT_VAL_3:
                    expval = expval * 10 + (c - '0');
                    break;
                case STORE_FEAT_VAL:
                    feat_val = (double)val;
                    expval = (expval * expsign) - decplaces;
                    feat_val = feat_val * pow(10, (double)expval);
                    if (vals_cursor >= vals_capacity) {
                        vals_capacity = 1 + (vals_cursor * 3 / 2);
                        realloc_vals = realloc(
                            vals, vals_capacity * sizeof(double));
                        if (realloc_vals == NULL) {
                            free(ys);
                            free(qids);
                            free(rows);
                            free(cols);
                            free(vals);
                            return PARSE_MEMORY_ERROR;
                        }
                        vals = realloc_vals;
                    }
                    vals[vals_cursor] = feat_val;
                    vals_cursor += 1;
                    break;
                default:
                    break;
            }

            // Get next state to go to given current state and char.
            char next_state = TRANSITIONS[current_state][c];

            // Return an appropriate error code if parsing fails due to invalid
            // format.
            if (next_state == INVALID) {
                free(ys);
                free(qids);
                free(rows);
                free(cols);
                free(vals);
                return PARSE_FORMAT_ERROR;
            }

            // Perform DFA state transition.
            current_state = next_state;
        }
    } while (bytes_read == BUFFER_SIZE);

    // If end of file is reached while parsing a feature value, finish processing it.
    if (current_state == PROCESS_FEAT_VAL_1 || current_state == PROCESS_FEAT_VAL_2 || current_state == PROCESS_FEAT_VAL_3) {
        feat_val = (double)val;
        expval = (expval * expsign) - decplaces;
        feat_val = feat_val * pow(10, (double)expval);
        if (vals_cursor >= vals_capacity) {
            vals_capacity = 1 + (vals_capacity * 3 / 2);
            realloc_vals = realloc(
                vals, vals_capacity * sizeof(double));
            if (realloc_vals == NULL) {
                free(ys);
                free(qids);
                free(rows);
                free(cols);
                free(vals);
                return PARSE_MEMORY_ERROR;
            }
            vals = realloc_vals;
        }
        vals[vals_cursor] = feat_val;
        vals_cursor += 1;
    }

    // Close file.
    fclose(fp);

    // Construct dense output matrix.
    double* xs = calloc((nr_cols - min_col) * row, sizeof(double));
    if (xs == NULL) {
        free(ys);
        free(qids);
        free(rows);
        free(cols);
        free(vals);
        return PARSE_MEMORY_ERROR;
    }
    for (size_t i=0; i<vals_cursor; i++) {
        xs[rows[i] * (nr_cols - min_col) + cols[i] - min_col] = vals[i];
    }

    // Free used resources.
    free(rows);
    free(cols);
    free(vals);

    // Shrink ys to correct size
    realloc_ys = realloc(ys, (1 + ys_cursor) * sizeof(int));
    if (realloc_ys == NULL) {
        free(ys);
        free(qids);
        return PARSE_MEMORY_ERROR;
    }
    ys = realloc_ys;

    // Shrink qids to correct size
    realloc_qids = realloc(qids, (1 + qids_cursor) * sizeof(long));
    if (realloc_qids == NULL) {
        free(ys);
        free(qids);
        return PARSE_MEMORY_ERROR;
    }
    qids = realloc_qids;

    // Set output variables
    *xs_out = xs;
    *ys_out = ys;
    *qids_out = qids;
    xs_shape->rows = row;
    xs_shape->cols = nr_cols - min_col;

    // Return success.
    return PARSE_OK;
}

#endif