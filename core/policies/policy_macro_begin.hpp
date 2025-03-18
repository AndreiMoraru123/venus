#define EnumValuePolicyObj(PolicyName, Ma, Mi, Val)                            \
  struct PolicyName : virtual public Ma {                                      \
    using MinorClass = typename Ma::Mi##TypeCate;                              \
    using Mi = typename Ma::Mi##TypeCate::Val;                                 \
  }

#define ValuePolicyObj(PolicyName, Ma, Mi, Val)                                \
  struct PolicyName : virtual public Ma {                                      \
    using MinorClass = Ma::Mi##ValueCate;                                      \
                                                                               \
  private:                                                                     \
    using type1 = decltype(Ma::Mi);                                            \
    using type2 = RemoveConstRef<type1>;                                       \
                                                                               \
  public:                                                                      \
    static constexpr type2 Mi = static_cast<type2>(Val);                       \
  }