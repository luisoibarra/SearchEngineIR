import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:ir_search_engine/controllers/option_controller.dart';
import 'package:ir_search_engine/widgets/option_item_widget.dart';

class OptionView extends StatelessWidget {
  const OptionView({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return GetX<OptionController>(
        init: OptionController(),
        builder: (controller) {
          return Flexible(
              child: Column(
            children: [
              OptionItemWidget(
                initialValue: controller.host.value,
                optionName: "Host",
                onChanged: (value) => controller.host.value = value,
              ),
              OptionItemWidget(
                initialValue: controller.port.value.toString(),
                optionName: "Port",
                onChanged: (value) =>
                    controller.port.value = int.tryParse(value) ?? 0,
              ),
              ElevatedButton(child: const Text("Save"), onPressed: () => controller.saveChanges())
            ],
          ));
        });
  }
}
