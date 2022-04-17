import 'package:flutter/material.dart';

class OptionItemWidget extends StatelessWidget {
  final String optionName;
  final Function(String newValue) onChanged;
  final String initialValue;

  const OptionItemWidget(
      {Key? key,
      required this.optionName,
      required this.initialValue,
      required this.onChanged})
      : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Card(
        child: Flexible(
      child: Row(
        children: [
          Text(optionName),
          SizedBox(width: 10),
          Flexible(
              child: TextFormField(
            initialValue: initialValue,
            onChanged: onChanged,
          ))
        ],
      ),
    ));
  }
}
