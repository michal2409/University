global _start

%define SYS_EXIT      60
%define SYS_OPEN      2
%define SYS_READ      0
%define SYS_CLOSE     3
%define O_RDONLY      0
%define BUFFER_SIZE   4096             ; bytes to read in one sys_read
%define MAGIC_CONST   68020            ; const used to code the message

section .data
fd dw 0                                ; file descriptor

section .bss
buffer resb BUFFER_SIZE                ; the bufferf or used to hold chunks read from file

section .text

; exiting program with code 0
exit_ok:
    mov rax, SYS_EXIT                  ; syscall code for exit
    xor rdi, rdi                       ; setting exit code to 0
    syscall

; exiting program with code 1
exit_error:
    mov rax, SYS_EXIT                  ; syscall code for exit
    mov rdi, 1                         ; setting exit code to 1
    syscall

; reads data from [fd] descriptor to buffer
read_to_buffer:
    mov rax, SYS_READ                  ; syscall code for reading file
    mov rdi, [fd]                      ; file descriptor
    mov rsi, buffer                    ; ptr to buffer where the data is stored
    mov rdx, BUFFER_SIZE               ; num of bytes to read
    syscall

    cmp rax, 0
    je check_and_finish                ; read 0 bytes, nothing more in file
    jl error_during_reading            ; negative number -> error during reading

    ; checking if number of read
    ; bytes is divisible by 4
    test rax, 1b
    jnz error_during_reading           ; if first bit is 1 then not divisible by 4
    test rax, 10b
    jnz error_during_reading           ; if second bit is 1 then not divisible by 4

    ret

; closes the [fd] descriptor
close_file:
    mov rax, SYS_CLOSE                 ; syscall code for closing file
    mov rdi, fd                        ; file descriptor
    syscall
    ret

; closing file and exiting with code 1
error_during_reading:
    call close_file
    call exit_error

; checking if sequence 6,8,0,2,0 occured
check_sequence:
    shl r14d, 4                        ; shifting by 4 bits to left
    and r14d, 0x0fffff                 ; keeping only lower 20 bits
    add r14d, r13d                     ; adding read number

    cmp r14d, 0x68020                  ; if number in register is 0x68020 it means there were sequence 6,8,0,2,0
    jne seq_not_found
    mov r8b, 1                         ; sequence found, setting flag to 1
    seq_not_found:
    ret

check_range:
    cmp r13d, MAGIC_CONST
    jle range_not_found                ; using <= for signed number
    mov r9b, 1
    range_not_found:
    ret

; checking flags and exiting with proper code
check_and_finish:
    call close_file

    cmp r10d, MAGIC_CONST              ; check if sum modulo 2^32 is 68020
    jne exit_error

    cmp r8b, 0                         ; check if sequence 6,8,0,2,0 occured
    je exit_error

    cmp r9b, 0                         ; check if there were number in range (68020, 2^31)
    je exit_error

    call exit_ok

_start:
    cmp qword [rsp], 2                 ; checking if argc == 2
    jne exit_error                     ; if argc != 2 exit with code 1

    mov rax, SYS_OPEN                  ; opening file
    mov rdi, [rsp + 16]                ; filename
    mov rsi, O_RDONLY                  ; read only flag
    syscall

    cmp rax, 0                         ; exit on error during opening the file
    jl exit_error
    mov [fd], rax                      ; save file descriptor for later use

    xor r8b, r8b                       ; flag for sequence 6,8,0,2,0
    xor r9b, r9b                       ; flag for number in (68020, 2^31)
    xor r10d, r10d                     ; for storing sum of numbers in file
    xor r14d, r14d                     ; for detecting 6,8,0,2,0

    read_chunk:
        call read_to_buffer

        mov r12, rax                   ; r12 stores the number of read bytes
        xor r11, r11                   ; r11 = 0

        process_chunk:                 ; for (r11 = 0; r11 < r12; ++r11)
            cmp r11, r12
            jb read_next_number
            call read_chunk            ; end of loop, read a new chunk of data
            read_next_number:

            xor r13d, r13d             ; r13d stores number from current iteration
            mov r13d, [buffer + r11]
            bswap r13d

            add r11, 4                 ; increment loop counter

            cmp r13d, MAGIC_CONST
            je error_during_reading    ; file contains MAGIC_CONST, no attack signal

            cmp r8b, 1
            je skip_checking_seq       ; already found sequence 6,8,0,2,0
            cmp r13d, 8
            ja zero_seq                ; if read number is greater than 8 then zeroing actual state
            call check_sequence
            jmp skip_checking_seq
            zero_seq:
            xor r14d, r14d
            skip_checking_seq:

            cmp r9b, 1
            je skip_checking_range     ; already found number in range (68020, 2^31)
            call check_range
            skip_checking_range:

            add r10d, r13d             ; storing sum of read numbers modulo 2^32

            jmp process_chunk
