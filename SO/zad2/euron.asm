global euron
extern get_value, put_value

section .data
NULL equ 0

section .bss
spinlock resb N*N    ; synchronizuje wymiane liczb przy operacji S
arr_num  resq N*N    ; przechowuje liczby do wymiany przy operacji S

section .text

; wstawia na stos liczbę od 0 do 9
push_digit:
  sub dl, '0'
  movzx rdx, dl
  push rdx
  jmp read_next_char

; zdejmuje dwie wartości ze stosu i wstawia ich sume
add_stack:
  pop r8
  add qword [rsp], r8
  jmp read_next_char

; zdejmuje dwie wartości ze stosu i wstawia ich iloczyn
mult_stack:
  pop  r8
  pop  r9
  imul r8, r9
  push r8
  jmp  read_next_char

; neguje arytmetycznie wartość na wierzchołku stosu
neg_stack:
  neg qword [rsp]
  jmp read_next_char

; wstawia na stos numer euronu
push_euron_id:
  push r12
  jmp  read_next_char

; zdejmuje wartość ze stosu ozn. w, jeśli teraz na wierzchu stosu
; jest wartość różna od 0 to przesuwa się o w operacji
branch:
  pop r8
  cmp qword [rsp], 0
  je  read_next_char
  add r13, r8
  jmp read_next_char

; zdejmuje wartość ze stosu
clean:
  pop r8
  jmp read_next_char

; duplikuje wartość na wierzchu stosu
duplicate:
  push qword [rsp]
  jmp  read_next_char

; zamienia miejscami dwie wartości na wierzchu stosu
exchange:
  pop  r8
  pop  r9
  push r8
  push r9
  jmp  read_next_char

; wstawia na stos wartość uzyskaną z
; wywołania funkcji get_value(uint64_t n)
push_get_value:
  mov  r14, rsp
  and  rsp, -16
  call get_value
  mov  rsp, r14
  push rax
  jmp  read_next_char

; zdejmuje wartość ze stosu ozn. w i wywołuje
; funkcję put_value(uint64_t n, uint64_t w)
call_put_value:
  mov  rdi, r12         ; ustawienie pierwszego arg na n
  pop  rsi              ; drugi argument to wartość ze stosu
  mov  r14, rsp
  and  rsp, -16
  call put_value
  mov  rsp, r14
  jmp  read_next_char

; zdejmuje wartość ze stosu, która traktowana jest jako numer euronu m,
; dalej czeka na operację 'S' euronu m ze zdjętym ze stosu numerem
; euronu n i zamienia wartości na wierzchołkach stosów euronów m i n
spin_lock:
  pop r8  ; numer euronu m

  ; zapisuje w r9 indeks euronu n, czyli n*N + m
  mov  r9, r12
  imul r9, N
  add  r9, r8

  ; zapisuje w r11 indeks euronu m, czyli m*N + n
  mov  r11, r8
  imul r11, N
  add  r11, r12

  write:
    cmp byte [spinlock + r11], 1
    je write ; euron m jeszcze nie odczytał liczby z tablicy

    pop qword [arr_num + r9*8]
    mov byte  [spinlock + r11], 1  ; sygnalizuje euronowi m, że wpisano liczbę to tablicy

  read:
    cmp byte [spinlock + r9], 0    ; czeka aż euron m wpisze liczbę do tablicy
    je read

    push qword [arr_num + r11*8]   ; wymiana liczb ze stosu z euronem m
    mov  byte  [spinlock + r9], 0  ; sygnalizuje, że euron n odczytał liczbę z tablicy

  jmp read_next_char

euron:
  push r12
  push r13
  push r14
  push rbp

  mov rbp, rsp
  mov r12, rdi
  mov r13, rsi
  mov dl, [r13]

  read_string:
    cmp dl, NULL  ; koniec napisu z operacjami
    je terminate

    cmp dl, 'B'
    je branch
    jl check_less_B

    ; większe od B
    cmp dl, 'G'
    je push_get_value
    jl check_less_G

    ; większe od S
    cmp dl, 'S'
    je spin_lock
    jl call_put_value
    jmp push_euron_id

    check_less_G:
      cmp dl, 'D'
      je duplicate
      jl clean
      jmp exchange

    check_less_B:
      cmp dl, '+'
      je add_stack
      jl mult_stack

      ; większe od +
      cmp dl, '-'
      je neg_stack
      jmp push_digit

    read_next_char:
      inc r13
      mov dl, [r13]

    jmp read_string

  terminate:
    pop rax
    mov rsp, rbp
    pop rbp
    pop r14
    pop r13
    pop r12

  ret
