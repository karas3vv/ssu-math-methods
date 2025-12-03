#import "conf.typ": conf, intro, conclusion
#show: conf.with(
  title: [ОТЧЕТ 
ПО ПРАКТИЧЕСКОЙ ПОДГОТОВКЕ

по дисциплине «Методы вычислений»],
  type: "no_type",
  info: (
    author: (
      name: [Карасева Вадима Дмитриевича],
      faculty: [КНиИТ],
      group: "351",
      sex: "male",
    ),
    inspector: (
      degree: "доцент, к. ф.-м. н.",
      name: "С. В. Миронов",
    ),
  ),
  settings: (
    title_page: (
      enabled: true,
    ),
    contents_page: (
      enabled: false,
    ),
  ),
)

//#intro
//#conclusion
#for value in ("tasks") {
  include "sections/" + value + ".typ"
}