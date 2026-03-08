Session 12.2 생성 완료. 주요 내용:

- **Bahdanau (Additive)**: $v^T \tanh(Ws + Uh)$ 수학적 정의, 이전 디코더 상태 사용
- **Luong (Multiplicative)**: Dot/General/Concat 세 가지 스코어 함수, 현재 디코더 상태 사용
- **Scaled Dot-Product**: 트랜스포머로 이어지는 진화, saturation 문제 해결
- **Mermaid 다이어그램 5개**: 계산 흐름, 스코어 함수 비교, 시퀀스 다이어그램, 진화 계보, 종합 비교
- **실행 가능한 코드**: BahdanauAttention, LuongDotAttention, LuongGeneralAttention 클래스 + 파라미터 수 비교
- **역사적 에피소드**: Bahdanau의 석사과정 시절 아이디어, Luong-Manning 협업, ICLR 리뷰 일화