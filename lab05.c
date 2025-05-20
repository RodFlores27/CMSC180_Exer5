#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <netdb.h>
#include <pthread.h>
#include <sched.h>
#include <math.h>

#define MAX_SLAVES 32
#define MAX_IP_LEN 64
#define BUFFER_SIZE 1024
#define CONFIG_FILE "config.txt"

// Structure to store slave information
typedef struct
{
    char ip[MAX_IP_LEN]; // IP address
    int port;            // port number
} SlaveInfo;

// Structure for thread arguments (used in core-affine version)
typedef struct
{
    int slave_idx;      // Index of the slave
    int n;              // Matrix size
    int **M;            // Matrix
    int *y;             // Vector y (NEW for Lab05)
    double *e;          // Vector e for results (NEW for Lab05)
    SlaveInfo slave;    // Slave information
    int cols_per_slave; // Number of columns per slave
    int num_slaves;     // Total number of slaves
} ThreadArgs;

// Function to read the configuration file
int read_config(char master_ip[MAX_IP_LEN], SlaveInfo slaves[], int *num_slaves, int is_slave)
{
    FILE *fp = fopen(CONFIG_FILE, "r");
    if (fp == NULL)
    {
        perror("Error opening config file");
        return -1;
    }

    char line[256];
    int slave_count = 0;

    while (fgets(line, sizeof(line), fp))
    {
        // Skip comments and empty lines
        if (line[0] == '#' || line[0] == '\n')
            continue;

        char ip[MAX_IP_LEN];
        int port;
        char role[10];

        if (sscanf(line, "%s %d %s", ip, &port, role) == 3)
        {
            if (strcmp(role, "master") == 0)
            {
                strcpy(master_ip, ip);
            }
            else if (strcmp(role, "slave") == 0)
            {
                if (!is_slave) // if master
                {
                    strcpy(slaves[slave_count].ip, ip);
                    slaves[slave_count].port = port;
                    slave_count++;
                }
            }
        }
    }

    fclose(fp);
    *num_slaves = slave_count;
    return 0;
}

// Function to compute Mean Square Error
// This is the computation that will happen on each slave
double compute_mse(int *x_col, int *y, int n)
{
    double sum = 0.0;
    for (int i = 0; i < n; i++)
    {
        double diff = (double)x_col[i] - (double)y[i];
        sum += diff * diff;
    }
    return sqrt(sum) / n;
}

// Function to run as master
int run_as_master(int n, int port, int num_slaves, SlaveInfo slaves[])
{
    printf("Running as master with n=%d, port=%d, slaves=%d\n", n, port, num_slaves);

    // Create a random n × n matrix M
    int **M = (int **)malloc(n * sizeof(int *));
    for (int i = 0; i < n; i++)
    {
        M[i] = (int *)malloc(n * sizeof(int));
        for (int j = 0; j < n; j++)
        {
            M[i][j] = (rand() % 9) + 1; // Random numbers from 1 to 9
        }
    }

    // Create a random vector y (NEW for Lab05)
    int *y = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++)
    {
        y[i] = (rand() % 9) + 1; // Random numbers from 1 to 9
    }

    // Create vector e to store final results (NEW for Lab05)
    double *e = (double *)malloc(n * sizeof(double));

    // Calculate columns per slave - we're now distributing by columns for MSE calculation
    int cols_per_slave = n / num_slaves;

    // Start timer - measures entire process including communication
    struct timespec time_before, time_after;
    clock_gettime(CLOCK_MONOTONIC, &time_before);

    // For each slave, create a socket, connect, and send data
    for (int s = 0; s < num_slaves; s++)
    {
        // Create socket
        int sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock < 0)
        {
            perror("Socket creation failed");
            continue;
        }

        // Set up server address
        struct sockaddr_in server_addr;
        memset(&server_addr, 0, sizeof(server_addr));
        server_addr.sin_family = AF_INET;
        server_addr.sin_port = htons(slaves[s].port);

        if (inet_pton(AF_INET, slaves[s].ip, &server_addr.sin_addr) <= 0)
        {
            perror("Invalid address");
            close(sock);
            continue;
        }

        // Connect to server
        if (connect(sock, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0)
        {
            perror("Connection failed");
            close(sock);
            continue;
        }

        printf("Connected to slave %d (%s:%d)\n", s, slaves[s].ip, slaves[s].port);

        // Send matrix dimensions
        send(sock, &n, sizeof(int), 0);

        // Send column start and count
        int start_col = s * cols_per_slave;
        int num_cols = (s == num_slaves - 1) ? (n - start_col) : cols_per_slave;
        send(sock, &start_col, sizeof(int), 0);
        send(sock, &num_cols, sizeof(int), 0);

        // REQUIREMENT 1: Send the full vector y (1MB - one-to-many broadcast)
        // Same vector y sent to all slaves
        send(sock, y, n * sizeof(int), 0);

        // Send matrix portion (columns)
        // We need to transpose the matrix for column-wise distribution
        for (int j = start_col; j < start_col + num_cols; j++)
        {
            // Extract column j from matrix M
            int col[n];
            for (int i = 0; i < n; i++)
            {
                col[i] = M[i][j];
            }
            // Send column
            send(sock, col, n * sizeof(int), 0);
        }

        // REQUIREMENT 3: Receive MSE results from slave (M1PR - many-to-one personalized reduction)
        // Each slave computes part of vector e and sends back only its portion
        double partial_e[num_cols];
        recv(sock, partial_e, num_cols * sizeof(double), 0);

        // Store received results in the appropriate portion of vector e
        for (int j = 0; j < num_cols; j++)
        {
            e[start_col + j] = partial_e[j];
        }

        close(sock);
    }

    // End timer
    clock_gettime(CLOCK_MONOTONIC, &time_after);
    double elapsed_time = (time_after.tv_sec - time_before.tv_sec) +
                          (time_after.tv_nsec - time_before.tv_nsec) / 1000000000.0;

    printf("\nMaster execution time: %0.9f seconds\n", elapsed_time);

    // Print a small portion of the results for verification
    printf("Mean Square Error Results (first 5 values):\n");
    for (int i = 0; i < (n < 5 ? n : 5); i++)
    {
        printf("e[%d] = %f\n", i, e[i]);
    }

    // Free memory
    for (int i = 0; i < n; i++)
    {
        free(M[i]);
    }
    free(M);
    free(y);
    free(e);

    return 0;
}

// Function to run as slave
int run_as_slave(int port, char master_ip[MAX_IP_LEN])
{
    printf("Running as slave with port=%d, master=%s\n", port, master_ip);

    // Set core affinity to a specific core
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset); // Use core 0
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

    // Create socket
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0)
    {
        perror("Socket creation failed");
        return -1;
    }

    // Set socket options to reuse address
    int opt = 1;
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0)
    {
        perror("Setsockopt failed");
        return -1;
    }

    // Bind socket to port
    struct sockaddr_in address;
    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port);

    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0)
    {
        perror("Bind failed");
        return -1;
    }

    // Start listening
    if (listen(server_fd, 3) < 0)
    {
        perror("Listen failed");
        return -1;
    }

    printf("Slave listening on port %d...\n", port);

    // Accept incoming connection
    struct sockaddr_in client_addr;
    socklen_t client_addr_len = sizeof(client_addr);

    int client_fd = accept(server_fd, (struct sockaddr *)&client_addr, &client_addr_len);
    if (client_fd < 0)
    {
        perror("Accept failed");
        return -1;
    }

    char client_ip[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &client_addr.sin_addr, client_ip, INET_ADDRSTRLEN);
    printf("Connection accepted from %s:%d\n", client_ip, ntohs(client_addr.sin_port));

    // Receive matrix dimensions
    int n;
    recv(client_fd, &n, sizeof(int), 0);

    // Receive column start and count
    int start_col, num_cols;
    recv(client_fd, &start_col, sizeof(int), 0);
    recv(client_fd, &num_cols, sizeof(int), 0);

    printf("Receiving submatrix: n=%d, start_col=%d, num_cols=%d\n", n, start_col, num_cols);

    // REQUIREMENT 1: Receive vector y (1MB - one-to-many broadcast)
    int *y = (int *)malloc(n * sizeof(int));
    recv(client_fd, y, n * sizeof(int), 0);

    // Allocate memory for column data
    int **columns = (int **)malloc(num_cols * sizeof(int *));
    for (int j = 0; j < num_cols; j++)
    {
        columns[j] = (int *)malloc(n * sizeof(int));
        recv(client_fd, columns[j], n * sizeof(int), 0);
    }

    // REQUIREMENT 2: Compute MSE for each column
    // Start computation timer - measures ONLY computation time
    struct timespec comp_time_before, comp_time_after;
    clock_gettime(CLOCK_MONOTONIC, &comp_time_before);

    // Calculate MSE for each column assigned to this slave
    double *partial_e = (double *)malloc(num_cols * sizeof(double));
    for (int j = 0; j < num_cols; j++)
    {
        partial_e[j] = compute_mse(columns[j], y, n);
    }

    // End computation timer
    clock_gettime(CLOCK_MONOTONIC, &comp_time_after);
    double comp_time = (comp_time_after.tv_sec - comp_time_before.tv_sec) +
                       (comp_time_after.tv_nsec - comp_time_before.tv_nsec) / 1000000000.0;

    printf("\nSlave computation time: %0.9f seconds\n", comp_time);

    // REQUIREMENT 3: Send partial results back to master (M1PR - many-to-one personalized reduction)
    send(client_fd, partial_e, num_cols * sizeof(double), 0);

    // Print a small portion of the results for verification
    printf("Computed MSE for columns %d to %d:\n", start_col, start_col + num_cols - 1);
    for (int j = 0; j < (num_cols < 5 ? num_cols : 5); j++)
    {
        printf("e[%d] = %f\n", start_col + j, partial_e[j]);
    }

    // Clean up
    for (int j = 0; j < num_cols; j++)
    {
        free(columns[j]);
    }
    free(columns);
    free(y);
    free(partial_e);
    close(client_fd);
    close(server_fd);

    return 0;
}

// Core-affine version of master function - using threads to manage connections
int run_as_master_core_affine(int n, int port, int num_slaves, SlaveInfo slaves[])
{
    printf("Running as master (core-affine) with n=%d, port=%d, slaves=%d\n", n, port, num_slaves);

    // Create a random n × n matrix M
    int **M = (int **)malloc(n * sizeof(int *));
    for (int i = 0; i < n; i++)
    {
        M[i] = (int *)malloc(n * sizeof(int));
        for (int j = 0; j < n; j++)
        {
            M[i][j] = (rand() % 9) + 1; // Random numbers from 1 to 9
        }
    }

    // Create a random vector y (NEW for Lab05)
    int *y = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++)
    {
        y[i] = (rand() % 9) + 1; // Random numbers from 1 to 9
    }

    // Create vector e to store final results (NEW for Lab05)
    double *e = (double *)malloc(n * sizeof(double));

    // Calculate columns per slave
    int cols_per_slave = n / num_slaves;

    // Start timer
    struct timespec time_before, time_after;
    clock_gettime(CLOCK_MONOTONIC, &time_before);

    // Thread function to handle slave communication
    void *slave_thread(void *arg)
    {
        ThreadArgs *args = (ThreadArgs *)arg;
        int s = args->slave_idx;
        int n = args->n;
        int **M = args->M;
        int *y = args->y;
        double *e = args->e;
        SlaveInfo slave = args->slave;
        int cols_per_slave = args->cols_per_slave;
        int num_slaves = args->num_slaves;

        // Set core affinity
        int max_cores = 11; // Adjust based on your machine
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(s % max_cores, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

        // Create socket
        int sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock < 0)
        {
            perror("Socket creation failed");
            pthread_exit(NULL);
        }

        // Set up server address
        struct sockaddr_in server_addr;
        memset(&server_addr, 0, sizeof(server_addr));
        server_addr.sin_family = AF_INET;
        server_addr.sin_port = htons(slave.port);

        if (inet_pton(AF_INET, slave.ip, &server_addr.sin_addr) <= 0)
        {
            perror("Invalid address");
            close(sock);
            pthread_exit(NULL);
        }

        // Connect to server
        if (connect(sock, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0)
        {
            perror("Connection failed");
            close(sock);
            pthread_exit(NULL);
        }

        printf("Thread %d connected to slave (%s:%d)\n", s, slave.ip, slave.port);

        // Send matrix dimensions
        send(sock, &n, sizeof(int), 0);

        // Send column start and count
        int start_col = s * cols_per_slave;
        int num_cols = (s == num_slaves - 1) ? (n - start_col) : cols_per_slave;
        send(sock, &start_col, sizeof(int), 0);
        send(sock, &num_cols, sizeof(int), 0);

        // REQUIREMENT 1: Send vector y (1MB - one-to-many broadcast)
        send(sock, y, n * sizeof(int), 0);

        // Send matrix portion (columns)
        for (int j = start_col; j < start_col + num_cols; j++)
        {
            // Extract column j
            int col[n];
            for (int i = 0; i < n; i++)
            {
                col[i] = M[i][j];
            }
            // Send column
            send(sock, col, n * sizeof(int), 0);
        }

        // REQUIREMENT 3: Receive MSE results from slave (M1PR - many-to-one personalized reduction)
        double partial_e[num_cols];
        recv(sock, partial_e, num_cols * sizeof(double), 0);

        // Store received results in the appropriate portion of vector e
        for (int j = 0; j < num_cols; j++)
        {
            e[start_col + j] = partial_e[j];
        }

        close(sock);
        pthread_exit(NULL);
    }

    // Create a thread for each slave
    pthread_t threads[MAX_SLAVES];
    ThreadArgs thread_args[MAX_SLAVES];

    for (int s = 0; s < num_slaves; s++)
    {
        thread_args[s].slave_idx = s;
        thread_args[s].n = n;
        thread_args[s].M = M;
        thread_args[s].y = y; // Pass vector y
        thread_args[s].e = e; // Pass vector e
        thread_args[s].slave = slaves[s];
        thread_args[s].cols_per_slave = cols_per_slave;
        thread_args[s].num_slaves = num_slaves;

        if (pthread_create(&threads[s], NULL, slave_thread, (void *)&thread_args[s]) != 0)
        {
            perror("Thread creation failed");
        }
    }

    // Wait for all threads to complete
    for (int s = 0; s < num_slaves; s++)
    {
        pthread_join(threads[s], NULL);
    }

    // End timer
    clock_gettime(CLOCK_MONOTONIC, &time_after);
    double elapsed_time = (time_after.tv_sec - time_before.tv_sec) +
                          (time_after.tv_nsec - time_before.tv_nsec) / 1000000000.0;

    printf("\nMaster execution time: %0.9f seconds\n", elapsed_time);

    // Print a small portion of the results for verification
    printf("Mean Square Error Results (first 5 values):\n");
    for (int i = 0; i < (n < 5 ? n : 5); i++)
    {
        printf("e[%d] = %f\n", i, e[i]);
    }

    // Free memory
    for (int i = 0; i < n; i++)
    {
        free(M[i]);
    }
    free(M);
    free(y);
    free(e);

    return 0;
}

int main(int argc, char *argv[])
{
    // Check command line arguments
    if (argc != 5)
    {
        printf("Usage: %s <n> <port> <status> <mode>\n", argv[0]);
        printf("  n: size of square matrix (for master), ignored for slave\n");
        printf("  port: port number to listen on\n");
        printf("  status: 0 for master, 1 for slave\n");
        printf("  mode: 0 for regular, 1 for core-affine\n");
        return 1;
    }

    int n = atoi(argv[1]);      // Matrix size
    int port = atoi(argv[2]);   // Port number
    int status = atoi(argv[3]); // Status (0 for master, 1 for slave)
    int mode = atoi(argv[4]);   // Mode (0 for regular, 1 for core-affine)

    // Seed random number generator
    srand(time(NULL));

    // Read configuration file
    SlaveInfo slaves[MAX_SLAVES];
    int num_slaves = 0;
    char master_ip[MAX_IP_LEN] = "";

    if (read_config(master_ip, slaves, &num_slaves, status) != 0)
    {
        return 1;
    }

    // Run as master or slave
    if (status == 0)
    {
        // Make sure we have slaves
        if (num_slaves == 0)
        {
            printf("No slaves found in configuration file\n");
            return 1;
        }
        printf("\nMaster IP: %s\n", master_ip);

        // Choose between regular and core-affine mode
        if (mode == 0)
            run_as_master(n, port, num_slaves, slaves);
        else
            run_as_master_core_affine(n, port, num_slaves, slaves);
    }
    else
    {
        // Make sure we have master IP
        if (strlen(master_ip) == 0)
        {
            printf("No master found in configuration file\n");
            return 1;
        }
        run_as_slave(port, master_ip);
    }

    return 0;
}